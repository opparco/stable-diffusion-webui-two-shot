# Latent Couple extension (two shot diffusion port)
This extension is an extension of the built-in Composable Diffusion.
This allows you to determine the region of the latent space that reflects your subprompts.

## Prerequisite
This extension need to apply cfg_denoised_callback-ea9bd9fc.patch (as of Feb 5, 2023 origin/HEAD commit ea9bd9fc).
```
git apply extensions/stable-diffusion-webui-two-shot/cfg_denoised_callback-ea9bd9fc.patch
```

## Extra generation params
Extra generation params provided by this extension are saved as PNG Info in the output file.
```
Latent Couple: "divisions=1:1,1:2,1:2 positions=0:0,0:0,0:1 weights=0.2,0.8,0.8 end at step=20"
```

# Credits
- two shot diffusion.ipynb https://colab.research.google.com/drive/1UdElpQfKFjY5luch9v_LlmSdH7AmeiDe?usp=sharing
