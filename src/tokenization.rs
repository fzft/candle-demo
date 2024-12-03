use tokenizers::{Tokenizer, TruncationParams, TruncationDirection, TruncationStrategy, Encoding};
use crate::error::TextEmbeddingsError;

pub struct Tokenization {
}

impl Tokenization {
    pub fn new() -> Self {
        Self { }     
    }

    pub fn encode(&self, tk: &mut Tokenizer, mut inputs: EncodingInput, add_special_tokens: bool, truncate: bool, max_input_length: usize) -> Result<Encoding, TextEmbeddingsError> {
        let truncate_params = truncate.then_some(TruncationParams{
            direction: TruncationDirection::Left,
            max_length: max_input_length,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        });
        match inputs {
            EncodingInput::Text(s) => {
                let encoding 
                = tk.with_truncation(truncate_params)?
                .encode(s, add_special_tokens)?;
                return Ok(encoding)
            }
            EncodingInput::Tokens(tokens) => {
                let encoding 
                = tk.with_truncation(truncate_params)?
                .encode(tokens, add_special_tokens)?;
                return Ok(encoding);
            }
        }
    }

    pub fn decode_ids(&self,tk: &mut Tokenizer, ids: Vec<u32>, skip_special_tokens: bool) -> Result<String, TextEmbeddingsError> {
        Ok(tk.with_truncation(None)?.decode(&ids, skip_special_tokens)?) 
    }
}


#[derive(Debug)]
pub enum EncodingInput {
    Text(String),
    Tokens(Vec<String>),
}

