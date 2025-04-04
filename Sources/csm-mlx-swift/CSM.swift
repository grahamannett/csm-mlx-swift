//
//  CSM.swift
//  csm-mlx-swift
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// MARK: - Language Model Components

private enum Language {
    // Specialized RMS Norm implementation
    class RMSNorm: Module, UnaryLayer {
        let weight: MLXArray
        let eps: Float

        public init(dimensions: Int, eps: Float = 1e-5) {
            self.weight = MLXArray.ones([dimensions]).asType(.float16)
            self.eps = eps
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            return MLXFast.rmsNorm(x, weight: 1.0 + self.weight, eps: self.eps)
        }
    }

    // Attention implementation for language models
    class Attention: Module {
        let config: BackboneConfiguration
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        let rope: RoPE

        public init(_ config: BackboneConfiguration) {
            self.config = config

            let dim = config.hiddenSize
            let heads = config.numAttentionHeads
            let kvHeads = config.numKeyValueHeads

            let headDim = config.headDim > 0 ? config.headDim : dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            self.rope = RoPE(dimensions: headDim, base: config.ropeTheta)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            let headDim =
                config.headDim > 0 ? config.headDim : config.hiddenSize / config.numAttentionHeads

            // prepare the queries, keys and values for the attention computation
            queries = queries.reshaped(B, L, config.numAttentionHeads, headDim).transposed(
                0, 2, 1, 3)
            keys = keys.reshaped(B, L, config.numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, config.numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

            if let cache {
                queries = rope(queries, offset: cache.offset)
                keys = rope(keys, offset: cache.offset)
                (keys, values) = cache.update(keys: keys, values: values)
            } else {
                queries = rope(queries)
                keys = rope(keys)
            }

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    // MLP implementation for language models
    class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(gelu(gate(x)) * up(x))
        }
    }

    // Transformer block implementation
    class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public init(_ config: BackboneConfiguration) {
            self._attention.wrappedValue = Attention(config)
            self.mlp = MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
        ) -> MLXArray {
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            let out = h + r
            return out
        }
    }

    // Implementation of a complete language model
    class LlamaModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        fileprivate let layers: [TransformerBlock]
        fileprivate let norm: RMSNorm

        let hiddenScale: Float

        public init(_ config: BackboneConfiguration) {
            precondition(config.vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

            self.hiddenScale = pow(Float(config.hiddenSize), 0.5)

            self.layers = (0..<config.numHiddenLayers)
                .map { _ in
                    TransformerBlock(config)
                }
            self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        public func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
            mask: MLXArray? = nil
        ) -> MLXArray {
            var h = inputEmbedding ?? embedTokens(inputs)
            h = h * hiddenScale

            let mask: MLXArray? =
                if mask == nil || (cache?[0].offset ?? 0) > 0 {
                    createAttentionMask(h: h, cache: cache)
                } else {
                    nil
                }

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return norm(h)
        }
    }

    // Language model wrapper that provides the logits and KV caches
    class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: LlamaModel

        var kvHeads: [Int]

        public init(_ config: BackboneConfiguration) {
            self.model = LlamaModel(config)

            self.kvHeads = (0..<config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        }

        // Added overload for DecoderConfiguration
        public init(_ config: DecoderConfiguration) {
            // Convert DecoderConfiguration to BackboneConfiguration
            let backboneConfig = BackboneConfiguration(
                modelType: config.modelType,
                vocabularySize: config.vocabularySize,
                numHiddenLayers: config.numHiddenLayers,
                numAttentionHeads: config.numAttentionHeads,
                numKeyValueHeads: config.numKeyValueHeads,
                headDim: config.headDim,
                hiddenSize: config.hiddenSize,
                maxSequenceLength: config.maxSequenceLength,
                intermediateSize: config.intermediateSize,
                rmsNormEps: config.rmsNormEps,
                ropeScaling: config.ropeScaling,
                ropeTheta: config.ropeTheta
            )
            self.model = LlamaModel(backboneConfig)
            self.kvHeads = (0..<config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        }

        public func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
            mask: MLXArray? = nil
        ) -> LMOutput {
            var out = model(inputs, cache: cache, inputEmbedding: inputEmbedding, mask: mask)
            out = model.embedTokens.asLinear(out)
            return LMOutput(logits: out)
        }
    }
}

// MARK: - CSM Model Implementation

/// CSM Model implementation
public class CSM: Module, LanguageModel, LoRAModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "backbone") private var backboneModel: Language.LanguageModel
    @ModuleInfo(key: "decoder") private var decoderModel: Language.LanguageModel

    @ModuleInfo(key: "text_embeddings") private var textEmbeddings: Embedding
    @ModuleInfo(key: "audio_embeddings") private var audioEmbeddings: Embedding
    @ModuleInfo(key: "projection") private var projection: Linear
    @ModuleInfo(key: "codebook0_head") private var codebook0Head: Linear
    @ModuleInfo(key: "audio_head") private var audioHead: MLXArray

    public let config: CSMConfiguration
    public var vocabSize: Int { config.textVocabSize }
    public var kvHeads: [Int] { backboneModel.kvHeads + decoderModel.kvHeads }

    public init(_ config: CSMConfiguration) {
        guard let backbone = config.backbone, let decoder = config.decoder else {
            fatalError("CSM requires both backbone and decoder configurations")
        }

        self.config = config

        // Calculate embedding dimensions
        let backboneEmbeddingDim =
            backbone.numAttentionHeads
            * (backbone.headDim > 0
                ? backbone.headDim : backbone.hiddenSize / backbone.numAttentionHeads)
        let decoderEmbeddingDim =
            decoder.numAttentionHeads
            * (decoder.headDim > 0
                ? decoder.headDim : decoder.hiddenSize / decoder.numAttentionHeads)

        // Initialize the language models
        self._backboneModel.wrappedValue = Language.LanguageModel(backbone)
        self._decoderModel.wrappedValue = Language.LanguageModel(decoder)

        // Initialize embeddings and projections
        self._textEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.textVocabSize,
            dimensions: backboneEmbeddingDim
        )

        self._audioEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.audioVocabSize * config.audioNumCodebooks,
            dimensions: backboneEmbeddingDim
        )

        self._projection.wrappedValue = Linear(
            backboneEmbeddingDim,
            decoderEmbeddingDim,
            bias: false
        )

        self._codebook0Head.wrappedValue = Linear(
            backboneEmbeddingDim,
            config.audioVocabSize,
            bias: false
        )

        // Initialize audio head
        self._audioHead.wrappedValue = MLXArray.zeros(
            [config.audioNumCodebooks - 1, decoderEmbeddingDim, config.audioVocabSize]
        )
    }

    // LanguageModel protocol conformance
    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        // Simplified implementation for now
        let tokens = input.text.tokens
        let logits = self.callAsFunction(tokens, cache: cache as? [KVCache])
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let logits = self.callAsFunction(input.tokens, cache: cache)
        return LMOutput(logits: logits)
    }

    // LoRAModel protocol conformance
    public func loraLinearLayers() -> MLXLMCommon.LoRALinearLayers {
        // TODO: look at lora stuff later
        // i think this should be both the backbone and the decoder?
        return backboneModel.model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }

    public func embedAudio(codebook: MLXArray, tokens: MLXArray) -> MLXArray {
        return audioEmbeddings(tokens + codebook * config.audioVocabSize)
    }

    public func embedTokens(tokens: MLXArray) -> MLXArray {
        // Implementation based on the provided Python code

        // 1. Process text tokens
        let textTokens = tokens[0..., 0..., -1]  // Select the last token ID (assumed text)
        var textEmbeds = self.textEmbeddings(textTokens)
        textEmbeds = expandedDimensions(textEmbeds, axis: -2)  // Add dimension: [B, S, 1, E]

        // 2. Process audio tokens
        let audioTokens = tokens[0..., 0..., ..<(-1)]  // Select all but the last token ID (assumed audio)

        // Create codebook indices and calculate offsets
        let codebookIndices = MLXArray(0..<config.audioNumCodebooks).asType(audioTokens.dtype)
        let codebookOffsets = codebookIndices * config.audioVocabSize

        // Add offsets to audio tokens (broadcasting applies)
        let adjustedAudioTokens = audioTokens + codebookOffsets

        // Flatten tokens for embedding lookup and get embeddings
        let flattenedAudioTokens = adjustedAudioTokens.flattened()
        var audioEmbeds = self.audioEmbeddings(flattenedAudioTokens)

        // Reshape audio embeddings back
        let B = tokens.dim(0)  // Batch size
        let S = tokens.dim(1)  // Sequence length
        audioEmbeds = audioEmbeds.reshaped(B, S, config.audioNumCodebooks, -1)  // Reshape to [B, S, C, E]

        // 3. Concatenate audio and text embeddings
        let combinedEmbeds = concatenated([audioEmbeds, textEmbeds], axis: -2)  // Concatenate along the codebook/text dimension

        return combinedEmbeds
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        // This is a placeholder implementation that needs to be refined based on the actual model architecture
        // In the actual model, this would:
        // 1. Embed tokens
        // 2. Pass them through the backbone model
        // 3. Project to decoder dimensions
        // 4. Pass through decoder model
        // 5. Generate output logits

        // TODO: THIS IS WRONG
        // let batchSize = inputs.shape[0]
        // let sequenceLength = inputs.shape[1]

        // // For now, just return zeros to represent logits
        // return zeros([batchSize, sequenceLength, config.textVocabSize])

        return zeros([inputs.dim(0), inputs.dim(1), config.textVocabSize])
    }

    // Provide a custom sanitize implementation
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        return weights  // No modifications needed yet
    }
}
