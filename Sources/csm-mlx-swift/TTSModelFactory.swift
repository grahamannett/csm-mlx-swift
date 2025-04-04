import Foundation
import Hub
import MLXLLM
import MLXLMCommon
import MLXVLM

public class TTSModelFactory: ModelFactory, @unchecked Sendable {
    public let typeRegistry: ModelTypeRegistry
    public let processorRegistry: ProcessorTypeRegistry
    public let modelRegistry: AbstractModelRegistry

    public init(
        typeRegistry: ModelTypeRegistry, processorRegistry: ProcessorTypeRegistry,
        modelRegistry: AbstractModelRegistry
    ) {
        self.typeRegistry = typeRegistry
        self.processorRegistry = processorRegistry
        self.modelRegistry = modelRegistry
    }

    public static let shared = LLMModelFactory(
        typeRegistry: LLMTypeRegistry.shared, modelRegistry: LLMRegistry.shared
    )

    public func _load(
        hub: HubApi, configuration: MLXLMCommon.ModelConfiguration,
        progressHandler: @escaping @Sendable (Progress) -> Void
    ) async throws -> MLXLMCommon.ModelContext {
        let modelDirectory = try await downloadModel(
            hub: hub,
            configuration: configuration,
            progressHandler: progressHandler
        )

        let configurationURL = modelDirectory.appending(component: "config.json")
        let configurationData = try Data(contentsOf: configurationURL)
        let baseConfig = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: configurationData
        )

        let model = try typeRegistry.createModel(
            configuration: configurationURL, modelType: baseConfig.modelType
        )

        // apply the weights to the bare model
        try loadWeights(
            modelDirectory: modelDirectory, model: model, quantization: baseConfig.quantization
        )

        let tokenizer = try await loadTokenizer(
            configuration: configuration,
            hub: hub
        )

        let processorConfiguration = modelDirectory.appending(
            component: "preprocessor_config.json"
        )
        let baseProcessorConfig: BaseProcessorConfiguration = try JSONDecoder().decode(
            BaseProcessorConfiguration.self,
            from: Data(
                contentsOf: processorConfiguration
            )
        )
        let processor = try processorRegistry.createModel(
            configuration: processorConfiguration,
            processorType: baseProcessorConfig.processorClass, tokenizer: tokenizer
        )

        return .init(
            configuration: configuration, model: model, processor: processor, tokenizer: tokenizer
        )
    }
}

public class TTSModelRegistry {
    public var registry: VLMRegistry

    public init() {
        self.registry = VLMRegistry()
    }

}
