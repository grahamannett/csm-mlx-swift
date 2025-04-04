#  csm swift


# notes
there are 2 csm versions to possibly use:
- https://github.com/SesameAILabs/csm
    - the original version
- https://github.com/senstella/csm-mlx
    - the forked version with mlx

using the original right now since that version is likely to be more current, but the mlx version is much better done




Current notes, I am still figuring out how to implement this as not used swift before:
- The README for the VLM models is much better than the `adding-model.md` file in the mlx-swift-examples
- It says to create a struct to match the config.json file, i believe that is the file from:
    - `.cache/huggingface/hub/models--sesame--csm-1b/snapshots/03ab46ff5cfdcc783cc76fcf9ea6fd0838503093/config.json`
    - but this file has like 5 keys while the examples always have like keys for many parameters
- Was trying to learn xcode and swift at the same time, but xcode is kinda dogsh*t.
    - everything is obfuscated behind menus/knowing where to click and i dont know any of the keyboard shortcuts (although the fold code animation is really cool)
*/
