# Discussion Notes with Han

## 10.12

### Model Architecture

- For generating input_embeds, we use BPE to encode the text. We use a VAE (frozen, not trainable) to generate image tokens from raw images, then apply a UnetDownsampler (conditional on timestep). Concatenate the image tokens with the text tokens, and we have the input_embeds.

- Seperate params (Q, K, V lists) in the transformer for text and image. 
  
### Training

- Use weighted loss from before diffusing and after diffusing.

- Maybe add processing for text/image token only scenario.