use fmmap::tokio::{AsyncMmapFileMut, AsyncMmapFileMutExt, AsyncOptions};
use fmmap::{MmapFileMut, MmapFileMutExt, MmapFileWriterExt, Options};
use futures::TryStreamExt;
use indicatif::ProgressBar;
use sqlx::mysql::{MySqlConnectOptions, MySqlPoolOptions};
use sqlx::{Executor, MySql, Row, Transaction};
use std::env;
use std::path::PathBuf;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), sqlx::Error> {
    const LIMIT: usize = 1_000_000;

    let connection_string = env::var("DB_CONNECTION_STRING")
        .expect("DB_CONNECTION_STRING environment variable was not set; expected `mysql://root:password@localhost/db`");
    let table = env::var("DB_TABLE").expect("DB_TABLE environment variable was not set");

    let pool = MySqlPoolOptions::new().connect(&connection_string).await?;
    let mut tx = pool.begin().await?;

    let num_vectors = get_num_vectors(&table, &mut tx).await?.min(LIMIT);
    let num_dimensions = get_num_floats(&table, &mut tx).await?;

    let pb = ProgressBar::new(num_vectors as u64);

    let path = PathBuf::from("vectors.bin");
    let options = AsyncOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .max_size((num_vectors * 4 * num_dimensions) as u64)
        .len(num_vectors * 4 * num_dimensions);

    let mut mmap = AsyncMmapFileMut::open_with_options(path, options)
        .await
        .unwrap();
    let mut writer = mmap.writer(0).unwrap();

    writer.write_u32(0).await.unwrap(); // version
    writer.write_u32(u32::MAX).await.unwrap(); // padding
    writer.write_u32(num_vectors as u32).await.unwrap();
    writer.write_u32(num_dimensions as u32).await.unwrap();
    writer.flush().await?;

    let query = format!(
        "SELECT `t`.`vector` FROM `{table}` AS `t` LIMIT {limit}",
        table = table,
        limit = LIMIT
    );
    let mut stream = sqlx::query(query.as_str()).fetch(&mut tx);
    while let Some(row) = stream.try_next().await? {
        let mut bytes: &[u8] = row.try_get(0)?;

        for _ in 0..num_dimensions {
            let float = bytes.read_f32_le().await?;
            writer.write_f32(float).await.unwrap();
        }

        pb.inc(1);
    }

    writer.flush().await?;
    pb.finish_and_clear();

    Ok(())
}

async fn get_num_vectors<'a>(
    table: &String,
    tx: &mut Transaction<'a, MySql>,
) -> Result<usize, sqlx::Error> {
    let query = format!("SELECT COUNT(*) FROM `{table}` AS `t`", table = table);
    let count: i64 = sqlx::query(query.as_str())
        .fetch_one(tx)
        .await?
        .try_get(0)?;
    Ok(count as usize)
}

async fn get_num_floats<'a>(
    table: &String,
    tx: &mut Transaction<'a, MySql>,
) -> Result<usize, sqlx::Error> {
    let query = format!(
        "SELECT `t`.`vector` FROM `{table}` AS `t` LIMIT 1",
        table = table
    );
    let row = sqlx::query(query.as_str()).fetch_one(tx).await?;
    let bytes: &[u8] = row.try_get(0)?;
    let num_floats = bytes.len() / 4;
    debug_assert_eq!(num_floats * 4, bytes.len());
    Ok(num_floats)
}
