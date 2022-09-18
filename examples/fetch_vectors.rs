use futures::TryStreamExt;
use sqlx::mysql::{MySqlConnectOptions, MySqlPoolOptions};
use sqlx::{Executor, Row};
use std::env;
use tokio::io::AsyncReadExt;

#[tokio::main]
async fn main() -> Result<(), sqlx::Error> {
    let connection_string = env::var("DB_CONNECTION_STRING")
        .expect("DB_CONNECTION_STRING environment variable was not set; expected `mysql://root:password@localhost/db`");
    let table = env::var("DB_TABLE").expect("DB_TABLE environment variable was not set");

    let pool = MySqlPoolOptions::new().connect(&connection_string).await?;

    let query = format!("SELECT `t`.`vector` FROM `{table}` AS `t`", table = table);
    let mut stream = sqlx::query(query.as_str()).fetch(&pool);
    while let Some(row) = stream.try_next().await? {
        let mut bytes: &[u8] = row.try_get(0)?;
        let num_floats = bytes.len() / 4;

        debug_assert_eq!(num_floats * 4, bytes.len());
        let mut floats = vec![0f32; num_floats];

        for float in floats.iter_mut() {
            *float = bytes.read_f32_le().await?;
        }

        let lol = floats;
    }

    Ok(())
}
