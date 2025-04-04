import Foundation
import MLX
import MLXLMCommon
import Hub

/// Helper functions for loading CSM models
public enum CSMLoader {
    /// Load a CSM model directly from configuration
    /// - Parameter config: The CSM configuration
    /// - Returns: A CSM model instance
    public static func loadModel(_ config: CSMConfiguration) -> CSM {
        return CSM(config)
    }

    /// Load a CSM model from config.json file
    /// - Parameter configURL: URL to the config.json file
    /// - Returns: A CSM model instance
    public static func loadModelFromConfig(_ configURL: URL) throws -> CSM {
        let configData = try Data(contentsOf: configURL)
        let csmConfig = try JSONDecoder().decode(CSMConfiguration.self, from: configData)
        return CSM(csmConfig)
    }
}