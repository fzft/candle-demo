use std::path::PathBuf;

use hf_hub::api::tokio::{ApiRepo, ApiError};

pub async fn download_artifacts(api: &ApiRepo) -> Result<PathBuf, ApiError> {
    download_pool_config(api).await?;
    download_config(api).await?;
    let tokenizer_path = download_tokenizer(api).await?;
    let model_path = tokenizer_path.parent().unwrap().to_path_buf();
    Ok(model_path)
}

pub async fn download_tokenizer(api: &ApiRepo) -> Result<PathBuf, ApiError>{
    println!("Downloading tokenizer");
    match api.get("tokenizer.json").await {
        Ok(p) => {
            println!("Downloaded tokenizer to {:?}", p);
            return Ok(p);
        }
        Err(e) => {
            return Err(e.into());
        }
    }
}

pub async fn download_config(api: &ApiRepo) -> Result<(), ApiError> {
    println!("Downloading config");
    match api.get("config.json").await {
        Ok(p) => {
            println!("Downloaded config to {:?}", p);
            return Ok(());
        }
        Err(e) => {
            return Err(e.into());
        }
    }
}

pub async fn download_pool_config(api: &ApiRepo) -> Result<(), ApiError> {
    println!("Downloading 1_pooling/config.json");
    match api.get("1_Pooling/config.json").await {
        Ok(p) => {
            println!("Downloaded pool config to {:?}", p);
            return Ok(());
        }
        Err(e) => {
            return Err(e.into());
        }
    }
}