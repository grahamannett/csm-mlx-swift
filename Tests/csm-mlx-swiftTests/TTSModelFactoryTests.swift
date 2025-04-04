import Foundation
import MLXVLM  // Add import for LLMRegistry
import Testing

@testable import csm_mlx_swift

@Test func testExtPackageRegistry() throws {
    // using this test to make sure the packages/modules are working from sweetpad
    // TODO: remove this print/test once i get the test targets working as I want them
    print("==> RUNNING TESTS FROM SWEETPAD")
    let modelConfig = MLXVLM.VLMRegistry.qwen2VL2BInstruct4Bit
    #expect(modelConfig != nil)
}

@Test func testLLMRegistrySharedConfigurations() throws {

}
