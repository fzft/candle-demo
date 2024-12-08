#[derive(Debug, thiserror::Error)]
pub enum TextEmbeddingsError {
    #[error("Tokenizer error: {0}")]
    TokenizerError(#[from] tokenizers::Error),
    // Add other error variants as needed
    #[error("Api error: {0}")]
    ApiError(#[from] hf_hub::api::tokio::ApiError),
}
