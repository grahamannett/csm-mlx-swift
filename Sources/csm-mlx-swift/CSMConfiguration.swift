import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Model Configuration

public struct TokenizerInfo: Codable, Sendable {
    public let repoId: String
    public let filename: String?

    public init(repoId: String, filename: String?) {
        self.repoId = repoId
        self.filename = filename
    }

    enum CodingKeys: String, CodingKey {
        case repoId = "repo_id"
        case filename
    }
}

// Common configurations for language models
public struct BackboneConfiguration: Codable, Sendable {
    public var modelType: String
    public var vocabularySize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var hiddenSize: Int
    public var maxSequenceLength: Int
    public var intermediateSize: Int
    public var rmsNormEps: Float
    public var ropeScaling: [String: StringOrNumber]?
    public var ropeTheta: Float

    public init(
        modelType: String = "llama",
        vocabularySize: Int = 128_256,
        numHiddenLayers: Int = 16,
        numAttentionHeads: Int = 32,
        numKeyValueHeads: Int = 8,
        headDim: Int = 64,
        hiddenSize: Int = 2048,
        maxSequenceLength: Int = 2048,
        intermediateSize: Int = 8192,
        rmsNormEps: Float = 1e-5,
        ropeScaling: [String: StringOrNumber]? = nil,
        ropeTheta: Float = 500_000.0
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.hiddenSize = hiddenSize
        self.maxSequenceLength = maxSequenceLength
        self.intermediateSize = intermediateSize
        self.rmsNormEps = rmsNormEps
        self.ropeScaling = ropeScaling
        self.ropeTheta = ropeTheta
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenSize = "hidden_size"
        case maxSequenceLength = "max_sequence_length"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeScaling = "rope_scaling"
        case ropeTheta = "rope_theta"
    }
}

public struct DecoderConfiguration: Codable, Sendable {
    public var modelType: String
    public var vocabularySize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var hiddenSize: Int
    public var maxSequenceLength: Int
    public var intermediateSize: Int
    public var rmsNormEps: Float
    public var ropeScaling: [String: StringOrNumber]?
    public var ropeTheta: Float

    public init(
        modelType: String = "llama",
        vocabularySize: Int = 128_256,
        numHiddenLayers: Int = 4,
        numAttentionHeads: Int = 8,
        numKeyValueHeads: Int = 2,
        headDim: Int = 128,
        hiddenSize: Int = 1024,
        maxSequenceLength: Int = 2048,
        intermediateSize: Int = 8192,
        rmsNormEps: Float = 1e-5,
        ropeScaling: [String: StringOrNumber]? = nil,
        ropeTheta: Float = 500_000.0
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.hiddenSize = hiddenSize
        self.maxSequenceLength = maxSequenceLength
        self.intermediateSize = intermediateSize
        self.rmsNormEps = rmsNormEps
        self.ropeScaling = ropeScaling
        self.ropeTheta = ropeTheta
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenSize = "hidden_size"
        case maxSequenceLength = "max_sequence_length"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeScaling = "rope_scaling"
        case ropeTheta = "rope_theta"
    }
}

public struct CSMConfiguration: Codable, Sendable {
    // public struct TokenizersDict: Codable, Sendable {
    //     public let audio: TokenizerInfo?
    //     public let text: TokenizerInfo?

    //     public init(audio: TokenizerInfo?, text: TokenizerInfo?) {
    //         self.audio = audio
    //         self.text = text
    //     }
    // }

    public let backbone: BackboneConfiguration?
    public let decoder: DecoderConfiguration?
    // public let tokenizers: TokenizersDict?

    public var modelType: String = "sesame/csm"
    public var audioNumCodebooks: Int = 32
    public var audioVocabSize: Int = 2051
    public var textVocabSize: Int = 128256

    public init(
        backbone: BackboneConfiguration? = nil,
        decoder: DecoderConfiguration? = nil,
        // tokenizers: TokenizersDict? = nil,
        modelType: String = "sesame/csm",
        audioNumCodebooks: Int = 32,
        audioVocabSize: Int = 2051,
        textVocabSize: Int = 128256
    ) {
        self.backbone = backbone
        self.decoder = decoder
        // self.tokenizers = tokenizers
        self.modelType = modelType
        self.audioNumCodebooks = audioNumCodebooks
        self.audioVocabSize = audioVocabSize
        self.textVocabSize = textVocabSize
    }

    enum CodingKeys: String, CodingKey {
        case backbone = "BACKBONE_CONFIGURATION"
        case decoder = "DECODER_CONFIGURATION"
        // case tokenizers = "TOKENIZERS"
        case modelType = "model_type"
        case audioNumCodebooks = "audio_num_codebooks"
        case audioVocabSize = "audio_vocab_size"
        case textVocabSize = "text_vocab_size"
    }
}
