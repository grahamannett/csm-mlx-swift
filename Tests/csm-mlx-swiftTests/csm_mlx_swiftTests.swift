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
        hiddenSize: 2048,
        intermediateSize: 8192,
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
        hiddenSize: 1024,
        intermediateSize: 8192,
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
}

@Test func testCSMModelConfiguration() throws {
    // Create a simple CSM configuration with minimal properties
    let config = CSMConfiguration(
        // Backbone configuration is now optional
        modelType: "sesame/csm",
        audioNumCodebooks: 32,
        audioVocabSize: 2051,
        textVocabSize: 128_256
    )

    // Verify the configuration properties
    #expect(config.modelType == "sesame/csm")
    #expect(config.audioNumCodebooks == 32)
    #expect(config.audioVocabSize == 2051)
    #expect(config.textVocabSize == 128_256)
}

@Test func testCSMBasicConfiguration() throws {
    // Create a simple CSM configuration without nested components
    let config = CSMConfiguration(
        modelType: "sesame/csm",
        audioNumCodebooks: 32,
        audioVocabSize: 2051,
        textVocabSize: 128_256
    )

    // Verify the configuration properties
    #expect(config.modelType == "sesame/csm")
    #expect(config.audioNumCodebooks == 32)
    #expect(config.audioVocabSize == 2051)
    #expect(config.textVocabSize == 128_256)
}
