import Foundation  // Needed for JSONDecoder if testing Codable
import Testing

@testable import csm_mlx_swift

@Test func configurationInitialization() throws {
    // 1. Create sample configurations
    let audioTokenizer = TokenizerInfo(
        repoId: "kyutai/moshiko-pytorch-bf16",
        filename: "tokenizer-e351c8d8-checkpoint125.safetensors"
    )

    let textTokenizer = TokenizerInfo(
        repoId: "unsloth/Llama-3.2-1B",
        filename: nil  // Should be valid for String?
    )

    // Access the nested struct correctly
    let tokenizersDict = CSMConfiguration.TokenizersDict(
        audio: audioTokenizer,
        text: textTokenizer
    )

    // Sample data based loosely on the Python "1b" example
    let backboneConfig = BackboneConfiguration(
        modelType: "llama",
        vocabularySize: 128_256,
        numHiddenLayers: 16,
        numAttentionHeads: 32,
        numKeyValueHeads: 8,
        headDim: 64,
        intermediateSize: 8192,
        hiddenSize: 2048,
        rmsNormEps: 1e-5,
        ropeScaling: [  // Use explicit enum cases
            "factor": StringOrNumber.float(32.0),
            "high_freq_factor": StringOrNumber.float(4.0),
            "low_freq_factor": StringOrNumber.float(1.0),
            "original_max_position_embeddings": StringOrNumber.int(8192),
            "rope_type": StringOrNumber.string("llama3"),
        ],
        ropeTheta: 500_000.0
    )

    // Sample data based loosely on the Python "100m" example
    let decoderConfig = DecoderConfiguration(
        modelType: "llama",
        vocabularySize: 128_256,
        numHiddenLayers: 4,
        numAttentionHeads: 8,
        numKeyValueHeads: 2,
        headDim: 128,
        intermediateSize: 8192,
        hiddenSize: 1024,
        rmsNormEps: 1e-5,
        ropeScaling: nil,  // Example with nil scaling
        ropeTheta: 500_000.0
    )

    // 2. Instantiate the main configuration
    let csmConfig = CSMConfiguration(
        backbone: backboneConfig,
        decoder: decoderConfig,
        tokenizers: tokenizersDict  // Pass the nested struct
    )

    // 3. Assertions using #expect
    #expect(csmConfig.backbone?.modelType == "llama")
    #expect(csmConfig.backbone?.hiddenSize == 2048)
    #expect(csmConfig.backbone?.ropeTheta == 500_000.0)
    #expect(csmConfig.backbone?.ropeScaling?.count == 5)  // Check dictionary content

    #expect(csmConfig.decoder?.modelType == "llama")
    #expect(csmConfig.decoder?.hiddenSize == 1024)
    #expect(csmConfig.decoder?.numHiddenLayers == 4)
    #expect(csmConfig.decoder?.ropeScaling == nil)

    #expect(csmConfig.tokenizers?.audio?.repoId == "kyutai/moshiko-pytorch-bf16")
    #expect(csmConfig.tokenizers?.audio?.filename == "tokenizer-e351c8d8-checkpoint125.safetensors")
    #expect(csmConfig.tokenizers?.text?.repoId == "unsloth/Llama-3.2-1B")
    #expect(csmConfig.tokenizers?.text?.filename == nil)

    // Optional: Test Codable conformance (Decoding)
    // Ensure the JSON string matches the Codable structure exactly
    /*
    let jsonString = """
    {
      "BACKBONE_CONFIGURATION": {
        "model_type": "llama",
        "vocab_size": 128256,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "intermediate_size": 8192,
        "hidden_size": 2048,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
          "factor": 32.0,
          "high_freq_factor": 4.0,
          "low_freq_factor": 1.0,
          "original_max_position_embeddings": 8192,
          "rope_type": "llama3"
        },
        "rope_theta": 500000.0
      },
      "DECODER_CONFIGURATION": {
        "model_type": "llama",
        "vocab_size": 128256,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 8192,
        "hidden_size": 1024,
        "rms_norm_eps": 1e-05,
        // NOTE: rope_scaling is intentionally omitted here to test nil decoding
        "rope_theta": 500000.0
      },
      "TOKENIZERS": {
        "audio": {
          "repo_id": "kyutai/moshiko-pytorch-bf16",
          "filename": "tokenizer-e351c8d8-checkpoint125.safetensors"
        },
        "text": {
          "repo_id": "unsloth/Llama-3.2-1B",
          "filename": null // Explicit null for optional field
        }
      }
    }
    """
    if let jsonData = jsonString.data(using: .utf8) {
        let decoder = JSONDecoder()
        do {
            let decodedConfig = try decoder.decode(CSMConfiguration.self, from: jsonData)
            #expect(decodedConfig.backbone?.hiddenSize == 2048)
            #expect(decodedConfig.tokenizers?.audio?.repoId == "kyutai/moshiko-pytorch-bf16")
            #expect(decodedConfig.decoder?.ropeScaling == nil)
            #expect(decodedConfig.tokenizers?.text?.filename == nil)
        } catch {
            #expect(Bool(false), "JSON Decoding failed: \(error)")
        }
    } else {
        #expect(Bool(false), "Failed to create data from JSON string")
    }
    */
}
