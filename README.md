# Noise Suppression model for Real-Life Human Audio

The goal was to create a noise suppression model for real-life human audio.

Check out the demonstration on [HuggingFace](https://huggingface.co/spaces/aryanthepain/Noise_Suppression_Model)

All the required libraries are in `requirements.txt` so that you can freely explore the project.
You can use the code below in your terminal to install the required libraries:

```bash
pip install -r ./requirements.txt
```

The file `final_approach.ipynb` contains the final approach to the problem solved using spectral subtraction which uses Short Time Fourier Transform(STFT) and Inverse STFT to complete the process.

We were going to solve this using an alternate initial approach using neural networks which is showcased in `alternate_failed_approach.ipynb`. However, the lack of results in the approach forced us to move to the final approach.

We hope all your noise problems are solved. :)

## Authors:-

- [Aryan Gupta](https://github.com/aryanthepain)
- [Vibha Gupta](https://github.com/Vibha17)
- [Tejas Deshmukh](https://github.com/tejas615)
- [Vaishnavi Agarwal](https://github.com/VA0910)
