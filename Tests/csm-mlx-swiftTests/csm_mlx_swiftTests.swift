import Foundation  // Needed for JSONDecoder if testing Codable
import Hub
import MLX
import MLXLMCommon
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

@Test func testCSMLoadModel() throws {
    // Create configurations for testing
    let backboneConfig = BackboneConfiguration(
        numHiddenLayers: 16,
        numAttentionHeads: 32,
        numKeyValueHeads: 8,
        hiddenSize: 2048,
        intermediateSize: 8192
    )

    let decoderConfig = DecoderConfiguration(
        numHiddenLayers: 4,
        numAttentionHeads: 8,
        numKeyValueHeads: 2,
        hiddenSize: 1024,
        intermediateSize: 4096
    )

    let config = CSMConfiguration(
        backbone: backboneConfig,
        decoder: decoderConfig,
        modelType: "sesame/csm",
        audioNumCodebooks: 32,
        audioVocabSize: 2051,
        textVocabSize: 128256
    )

    // Create model directly
    let model = CSM(config)

    // Assert model was created with expected properties
    #expect(model.config.modelType == "sesame/csm")
    #expect(model.config.backbone?.numHiddenLayers == 16)
    #expect(model.config.decoder?.numHiddenLayers == 4)
    #expect(model.config.audioNumCodebooks == 32)
}

@Test func testCSMDirectLoading() throws {
    // Create a test directory
    let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(
        "csm-test-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
        at: tempDir, withIntermediateDirectories: true)

    // Create test configs
    let backboneConfig = BackboneConfiguration(
        numHiddenLayers: 16,
        numAttentionHeads: 32,
        numKeyValueHeads: 8,
        hiddenSize: 2048,
        intermediateSize: 8192
    )

    let decoderConfig = DecoderConfiguration(
        numHiddenLayers: 4,
        numAttentionHeads: 8,
        numKeyValueHeads: 2,
        hiddenSize: 1024,
        intermediateSize: 4096
    )

    let config = CSMConfiguration(
        backbone: backboneConfig,
        decoder: decoderConfig,
        modelType: "sesame/csm",
        audioNumCodebooks: 32,
        audioVocabSize: 2051,
        textVocabSize: 128256
    )

    // Create the config.json file
    let configData = try JSONEncoder().encode(config)
    let configPath = tempDir.appendingPathComponent("config.json")
    try configData.write(to: configPath)

    // Create additional base configuration for model type
    let baseConfig = ["model_type": "sesame/csm", "quantization": "none"]
    let baseConfigData = try JSONSerialization.data(withJSONObject: baseConfig)
    let baseConfigPath = tempDir.appendingPathComponent("base_config.json")
    try baseConfigData.write(to: baseConfigPath)

    // Create an empty weights file
    let weightsDir = tempDir.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: weightsDir, withIntermediateDirectories: true)
    let emptyWeights = Data()
    try emptyWeights.write(to: weightsDir.appendingPathComponent("model.safetensors"))

    // Note: We can't actually load the model from disk in this test
    // since we don't have real weights, but we can verify the model
    // can be created directly
    let model = CSM(config)

    #expect(model.config.modelType == "sesame/csm")
    #expect(model.config.backbone?.numHiddenLayers == 16)
    #expect(model.config.decoder?.numHiddenLayers == 4)
}

@Test func testCSMLoader() throws {
    // Create test configs
    let backboneConfig = BackboneConfiguration(
        numHiddenLayers: 16,
        numAttentionHeads: 32,
        numKeyValueHeads: 8,
        hiddenSize: 2048,
        intermediateSize: 8192
    )

    let decoderConfig = DecoderConfiguration(
        numHiddenLayers: 4,
        numAttentionHeads: 8,
        numKeyValueHeads: 2,
        hiddenSize: 1024,
        intermediateSize: 4096
    )

    let config = CSMConfiguration(
        backbone: backboneConfig,
        decoder: decoderConfig,
        modelType: "sesame/csm",
        audioNumCodebooks: 32,
        audioVocabSize: 2051,
        textVocabSize: 128256
    )

    // Test direct creation
    let model = CSM(config)

    // Assert model was created with expected properties
    #expect(model.config.modelType == "sesame/csm")
    #expect(model.config.backbone?.numHiddenLayers == 16)
    #expect(model.config.decoder?.numHiddenLayers == 4)
    #expect(model.config.audioNumCodebooks == 32)

    // Test loading from config file
    let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(
        "csm-test-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(
        at: tempDir, withIntermediateDirectories: true)

    let configData = try JSONEncoder().encode(config)
    let configPath = tempDir.appendingPathComponent("config.json")
    try configData.write(to: configPath)

    // Load the config again and create a new model
    let loadedConfigData = try Data(contentsOf: configPath)
    let loadedConfig = try JSONDecoder().decode(CSMConfiguration.self, from: loadedConfigData)
    let loadedModel = CSM(loadedConfig)

    #expect(loadedModel.config.modelType == "sesame/csm")
    #expect(loadedModel.config.backbone?.numHiddenLayers == 16)
    #expect(loadedModel.config.decoder?.numHiddenLayers == 4)
    #expect(loadedModel.config.audioNumCodebooks == 32)
}