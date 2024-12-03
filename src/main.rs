mod download;
mod tokenization;
mod infer;
mod error;

use anyhow::Result;
use crate::download::{download_artifacts};
use crate::tokenization::{Tokenization, EncodingInput};
use crate::error::TextEmbeddingsError;
use tokenizers::Tokenizer;

#[tokio::main]
async fn main() -> Result<(), TextEmbeddingsError> {
    let api = hf_hub::api::tokio::Api::new().unwrap();
    let api_repo = api.model("BAAI/bge-m3".to_string());
    let model_path = download_artifacts(&api_repo).await?;
    let mut tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))?;
    let tokenization = Tokenization::new();
    let encoded = tokenization.encode(&mut tokenizer, EncodingInput::Text("Hello, world!".to_string()) , true, true, 512)?;
    println!("tokens: {:?}", encoded.get_tokens());
    println!("ids: {:?}", encoded.get_ids());
    println!("vocab size: {:?}", tokenizer.get_vocab_size(true));

    let encoded2 = tokenizer.encode("你好， 很高兴见到你", true)?;
    println!("tokens: {:?}", encoded2.get_tokens());
    println!("ids: {:?}", encoded2.get_ids());

    let decoded = tokenization.decode_ids(&mut tokenizer, encoded2.get_ids().to_vec(), true)?;
    println!("decoded: {:?}", decoded);

    Ok(())
}


