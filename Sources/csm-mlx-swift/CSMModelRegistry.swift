import Foundation
import Hub
import MLX
import MLXLMCommon

/// Registry of CSM model type
public class CSMTypeRegistry: ModelTypeRegistry, @unchecked Sendable {
    /// Shared instance with default model types
    public static let shared: CSMTypeRegistry = .init(creators: all())

    /// All predefined model types
    private static func all() -> [String: @Sendable (URL) throws -> any LanguageModel] {
        [
            "sesame/csm": create(CSMConfiguration.self, CSM.init)
        ]
    }
}

/// Registry of models and any overrides that go with them
public class CSMRegistry: AbstractModelRegistry, @unchecked Sendable {
    /// Shared instance with default model configurations
    public static let shared: CSMRegistry = .init(modelConfigurations: all())

    static public let csm1b = ModelConfiguration(
        id: "sesame/csm-1b",
        defaultPrompt: "Hello from Sesame."
    )

    static private func all() -> [ModelConfiguration] {
        [
            csm1b
        ]
    }
}

/// Factory for creating new CSM models
public class CSMModelFactory: ModelFactory, @unchecked Sendable {
    public init(
        typeRegistry: ModelTypeRegistry,
        modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

    /// Shared instance with default behavior
    public static let shared = CSMModelFactory(
        typeRegistry: CSMTypeRegistry.shared,
        modelRegistry: CSMRegistry.shared
    )

    /// Registry of model type
    public let typeRegistry: ModelTypeRegistry

    /// Registry of model id to configuration
    public let modelRegistry: AbstractModelRegistry

    public func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContext {
        // Download weights and config
        let modelDirectory = try await downloadModel(
            hub: hub, configuration: configuration, progressHandler: progressHandler)

        // Load the configuration
        let configurationURL = modelDirectory.appending(component: "config.json")
        let baseConfig = try JSONDecoder().decode(
            BaseConfiguration.self, from: Data(contentsOf: configurationURL))

        // Create the model
        let model = try typeRegistry.createModel(
            configuration: configurationURL, modelType: baseConfig.modelType)

        // Apply weights to the model
        try loadWeights(
            modelDirectory: modelDirectory, model: model, quantization: baseConfig.quantization)

        // Load tokenizer
        let tokenizer = try await loadTokenizer(
            configuration: configuration, hub: hub)

        // No processor for CSM model yet
        let dummyProcessor = DummyProcessor()

        return .init(
            configuration: configuration, model: model, processor: dummyProcessor,
            tokenizer: tokenizer)
    }
}

/// Dummy processor until we have a proper CSM processor
private class DummyProcessor: UserInputProcessor {
    public func prepare(input: UserInput) throws -> LMInput {
        let promptTokens = [0]  // Just a placeholder
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        return LMInput(text: .init(tokens: promptArray))
    }
}
