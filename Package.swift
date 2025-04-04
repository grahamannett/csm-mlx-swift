// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "csm-mlx-swift",
    platforms: [
        .macOS("14.0"),
        .iOS(.v16),
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "csm-mlx-swift",
            targets: ["csm-mlx-swift"]),
    ],
    dependencies:[
        .package(url: "https://github.com/ml-explore/mlx-swift-examples/", branch: "main"),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.21.2")),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "0.1.17")
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "csm-mlx-swift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
                .product(name: "MLXVLM", package: "mlx-swift-examples"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "csm-mlx-swiftTests",
            dependencies: ["csm-mlx-swift"]
        ),
       .testTarget( // if possible, get multiple targets to run from sweetpad?
           name: "TTSModelFactoryTests",
           dependencies: [ "csm-mlx-swift"]
       ),
    ]
)
