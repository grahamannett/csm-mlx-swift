//
//  File.swift
//  csm-mlx-swift
//

/*

Note: I am still figuring out how to implement this as not used swift before.

Some notes:
- The `adding-model.md` file in the mlx-swift-examples is kinda useless since doesnt really explain how to add models that may utilize other models
- It says to create a struct to match the config.json file, i believe that is the file from:
    - `.cache/huggingface/hub/models--sesame--csm-1b/snapshots/03ab46ff5cfdcc783cc76fcf9ea6fd0838503093/config.json`
    - but this file has like 5 keys while the examples always have like keys for many parameters

- Was trying to learn xcode and swift at the same time, but xcode is kinda dogshit.
    - everything is obfuscated behind menus/knowing where to click and all the keyboard shortcuts are fucked (although the fold code animation is really cool)

*/

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN


public struct CSMConfiguration: Codable, Sendable {



    // this is the "llama-1B" in `FLAVORS`
    public struct BackboneConfiguration: Codable, Sendable {
        var modelType: String = "llama"
        var vocabularySize: Int = 128_256
        var numHiddenLayers: Int = 16
        var numAttentionHeads: Int = 32
        var numKeyValueHeads: Int = 8
        var hiddenSize: Int = 2048
        var maxSequenceLength: Int = 2048
        var intermediateSize: Int = 8192
        var attentionDropout: Float = 0.0
        var rmsNormEps: Float = 1e-5
        var ropeTheta: Float = 500_000.0
        var ropeScalingFactor: Float = 32.0

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case hiddenSize = "hidden_size"
            case maxSequenceLength = "max_sequence_length"
            case intermediateSize = "intermediate_size"
            case attentionDropout = "attention_dropout"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case ropeScalingFactor = "rope_scaling_factor"
        }
    }

    // this is the "llama-100M" in `FLAVORS`
    public struct DecoderConfiguration: Codable, Sendable {
        var modelType: String = "llama"
        var vocabularySize: Int = 128_256 // does _ work for swift ints?
        var numHiddenLayers: Int = 4
        var numAttentionHeads: Int = 8
        var numKeyValueHeads: Int = 2
        var hiddenSize: Int = 1024
        var maxSequenceLength: Int = 2048
        var intermediateSize: Int = 8192
        var attentionDropout: Float = 0.0
        var rmsNormEps: Float = 1e-5
        var ropeTheta: Float = 500_000.0
        var ropeScalingFactor: Float = 32.0

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case hiddenSize = "hidden_size"
            case maxSequenceLength = "max_sequence_length"
            case intermediateSize = "intermediate_size"
            case attentionDropout = "attention_dropout"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case ropeScalingFactor = "rope_scaling_factor"
        }
    }


    var modelType: String = "sesame/csm"
    var audioNumCodebooks: Int = 32
    var audioVocabSize: Int = 2051
    var backboneFlavor: String = "llama-1B"
    var decoderFlavor: String = "llama-100M"
    var textVocabSize: Int = 128256

    public let backboneConfiguration: BackboneConfiguration
    public let decoderConfiguration: DecoderConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case audioNumCodebooks = "audio_num_codebooks"
        case audioVocabSize = "audio_vocab_size"
        case backboneFlavor = "backbone_flavor"
        case decoderFlavor = "decoder_flavor"
        case textVocabSize = "text_vocab_size"
    }
}

// looking at PaliGemma seems like the most helpful way to understand how to implement this
public class CSM: Module, LLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "backbone") private var backboneModel: Language.LanguageModel
    @ModuleInfo(key: "decoder") private var decoderModel: Language.LanguageModel

    public let config: CSMConfiguration
    public var vocabSize: Int { config.textVocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }


    public init(_ config: CSMConfiguration) {
        self.config = config
        self.backbone.wrappedValue = Language.LanguageModel(config.backboneConfiguration)
        self.decoder.wrappedValue = Language.LanguageModel(config.decoderConfiguration)
    }
}
