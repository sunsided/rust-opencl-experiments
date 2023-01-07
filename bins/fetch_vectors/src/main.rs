use fmmap::tokio::{AsyncMmapFileMut, AsyncMmapFileMutExt, AsyncOptions};
use futures::TryStreamExt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use sqlx::{mysql::MySqlPoolOptions, MySql, Row, Transaction};
use std::env;
use std::path::PathBuf;
use tokio::io;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::task::JoinHandle;

#[tokio::main]
async fn main() -> Result<(), sqlx::Error> {
    const LIMIT: usize = 1_000_000;
    const HEADER_SIZE: usize = 16;

    dotenvy::dotenv().ok();

    let connection_string = env::var("DB_CONNECTION_STRING")
        .expect("DB_CONNECTION_STRING environment variable was not set; expected `mysql://root:password@localhost/db`");
    let table = env::var("DB_TABLE").expect("DB_TABLE environment variable was not set");

    let pool = MySqlPoolOptions::new().connect(&connection_string).await?;
    let mut tx = pool.begin().await?;

    let num_vectors = get_num_vectors(&table, &mut tx).await?.min(LIMIT);
    let num_dimensions = get_num_floats(&table, &mut tx).await?;

    let pb_r = ProgressBar::new(num_vectors as u64);
    pb_r.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} Fetch {pos}/{len} {elapsed_precise} [{wide_bar:.cyan/blue}] {eta}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let pb_w = ProgressBar::new(num_vectors as u64);
    pb_w.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} Store {pos}/{len} {elapsed_precise} [{wide_bar:.cyan/blue}] {eta}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let mp = MultiProgress::new();
    let pb_r = mp.add(pb_r);
    let pb_w = mp.add(pb_w);

    let path = PathBuf::from("../../vectors.bin");

    let options = AsyncOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .max_size((num_vectors * 4 * num_dimensions + HEADER_SIZE) as u64)
        .len(num_vectors * 4 * num_dimensions + HEADER_SIZE);

    let mut mmap = AsyncMmapFileMut::open_with_options(path, options)
        .await
        .unwrap();

    let (sender, mut recv) = tokio::sync::mpsc::unbounded_channel();

    let write: JoinHandle<io::Result<()>> = tokio::spawn(async move {
        let mut writer = mmap.writer(0).unwrap();

        writer.write_u32(0).await.unwrap(); // version
        writer.write_u32(u32::MAX).await.unwrap(); // padding
        writer.write_u32(num_vectors as u32).await.unwrap();
        writer.write_u32(num_dimensions as u32).await.unwrap();
        writer.flush().await?;

        while let Some(vec) = recv.recv().await {
            for float in vec {
                writer.write_f32(float).await?;
            }

            pb_w.inc(1);
        }

        writer.flush().await?;
        pb_w.finish_and_clear();
        Ok(())
    });

    let read: JoinHandle<Result<(), sqlx::Error>> = tokio::spawn(async move {
        let query = format!(
            "SELECT `t`.`vector` FROM `{table}` AS `t` ORDER BY `internal_id` ASC LIMIT {limit}",
            table = table,
            limit = LIMIT
        );

        let mut stream = sqlx::query(query.as_str()).fetch(&mut tx);
        while let Some(row) = stream.try_next().await? {
            let mut bytes: &[u8] = row.try_get(0)?;

            let mut vec = Vec::with_capacity(num_dimensions);
            for _ in 0..num_dimensions {
                let float = bytes.read_f32_le().await?;
                vec.push(float);
            }

            sender.send(vec).unwrap();
            pb_r.inc(1);
        }

        pb_r.finish_and_clear();
        Ok(())
    });

    let _ = tokio::join!(write, read);

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
