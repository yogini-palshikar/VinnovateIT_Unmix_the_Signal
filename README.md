# Unmix the Signal: Blind Audio Source Separation with ICA

This project takes three audio sources---speech, music and ambient
sound---mixes them together, then tries to recover each source using
Independent Component Analysis (ICA). The main pipeline is built around
NumPy, with centering, PCA whitening and a symmetric FastICA
implementation.

The current run uses real recordings placed in `data/sources/`. It also
includes a few extra experiments: a comparison with PCA, a noise-level
sweep, and a small audio-fingerprinting demo.

## What the project does

The goal is to recover the original sources from mixtures of the form

\[ X = AS \]

where `S` contains the original sources, `A` is the mixing matrix, and
`X` contains the observed mixtures. The separation algorithm is not
given the source-to-mixture assignments in advance. In this controlled
experiment, the mixing matrix is used to create the mixtures and to
evaluate the recovered signals.

The main steps are:

1.  Load the source recordings and prepare them for processing.
2.  Convert the audio to mono, resample it to the target sample rate,
    crop it to the requested duration, and normalise it.
3.  Mix the sources using the matrix in `config.py`.
4.  Centre the mixtures and whiten them using PCA.
5.  Run FastICA to estimate the independent components.
6.  Match the estimated components to the reference sources for
    evaluation.
7.  Save the separated audio, plots and metrics.
8.  Run the optional bonus experiments unless `--no-bonus` is supplied.

ICA can recover sources only under suitable conditions. In particular,
the sources should be sufficiently independent, the mixing should be
approximately linear and instantaneous, and the number of mixtures
should be at least the number of sources.

## Features

-   FastICA implemented from scratch, including symmetric decorrelation
    and configurable nonlinearities (`tanh`, `cube` and `gauss`).
-   PCA whitening using an eigenvalue decomposition, with checks for
    degenerate input.
-   Evaluation using source/component matching, Pearson correlation,
    normalised mean squared error (NMSE), mean absolute error (MAE),
    SI-SDR and the Amari index.
-   WAV exports for listening to the mixtures and recovered sources.
-   Plots for waveforms, covariance before and after whitening,
    estimated components, spectrograms, correlation, metrics and
    convergence.
-   CSV and JSON files containing the main metrics and bonus results.
-   Three optional experiments: PCA vs. ICA, an input-SNR sweep, and a
    mini-Shazam fingerprinting demo.

## Requirements

The project was run with Python 3.12 and uses the packages listed in
`requirements.txt`.

## Setup and run

Open a terminal in the project folder and activate your virtual
environment if you use one.

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put the three source recordings in `data/sources/` using these exact
filenames:

``` text
data/sources/
├── speech.wav
├── music.wav
└── ambient.wav
```

Then run the full pipeline:

``` powershell
python main.py
```

To run only the main separation pipeline and skip the bonus experiments:

``` powershell
python main.py --no-bonus
```

Optional command-line arguments:

``` powershell
python main.py --duration 15 --sr 16000 --seed 0
```

-   `--duration`: length of each source segment in seconds.
-   `--sr`: target sample rate in Hz.
-   `--seed`: random seed used by the pipeline.

The current run used a duration of 15 seconds, a sample rate of 16 kHz
and seed `0`.

If the three real recordings are not present, the project can use its
synthetic-source fallback. The generated results identify whether the
run used real or synthetic sources.

## Results from the current real-recording run

These values were printed by `python main.py` using the source
recordings currently in `data/sources/`. They are results from this
particular run, not a guarantee that every set of recordings will
produce the same scores.

### Main separation

FastICA converged in 5 iterations. The reported Amari index was
`0.0033`, the mixing matrix condition number was `2.92`, and the maximum
absolute deviation of the whitened covariance from the identity matrix
was `4.4e-15`.

  ------------------------------------------------------------------------------------
  Source         Matched  Pearson r   Absolute r         NMSE          MAE SI-SDR (dB)
               component                                                   
  ---------- ----------- ---------- ------------ ------------ ------------ -----------
  Speech               2    +1.0000       1.0000       0.0000       0.0012       55.89

  Music                0    +1.0000       1.0000       0.0001       0.0065       40.25

  Ambient              1    +1.0000       1.0000       0.0000       0.0027       46.82

  **Mean**           ---        ---   **1.0000**   **0.0000**   **0.0035**   **47.65**
  ------------------------------------------------------------------------------------

The correlation values are rounded to four decimal places. Since the
scores are unusually high, it is worth listening to the separated files
and checking the matching/evaluation code when using a new set of
recordings. The reference sources are used to score the output; they are
not inputs to FastICA itself.

### Bonus A: PCA vs. ICA

All methods were evaluated on the same mixtures using the same
source-matching procedure.

  Method                           Mean absolute correlation   Mean SI-SDR (dB)   Mean NMSE
  ------------------------------ --------------------------- ------------------ -----------
  Raw mixtures (no processing)                        0.9049               6.74      0.1804
  PCA (no whitening)                                  0.6848              -0.41      0.5191
  PCA-whitened                                        0.6848              -0.41      0.5191
  FastICA                                             1.0000              47.65      0.0000

PCA is useful for decorrelation and whitening, but it does not generally
separate sources by statistical independence on its own. These figures
describe this experiment and should not be read as a universal ranking
for every dataset.

### Bonus B: Input SNR sweep

The experiment adds noise to the mixtures at several signal-to-noise
ratio (SNR) levels. Each row is the mean of three noise trials.

  --------------------------------------------------------------------------
  Input SNR (dB)   Measured SNR    Mean SI-SDR  Mean absolute      Converged
                           (dB)           (dB)    correlation         trials
  -------------- -------------- -------------- -------------- --------------
              30          30.00   26.89 ± 0.01         0.9990            3/3

              20          20.00   16.98 ± 0.01         0.9900            3/3

              10           9.99    7.17 ± 0.01         0.9152            3/3

               0          -0.00   -2.04 ± 0.01         0.6201            3/3
  --------------------------------------------------------------------------

As expected, separation quality falls as the input becomes noisier.
FastICA converged in all trials shown here, although convergence does
not by itself mean that the recovered sources are accurate.

### Bonus C: Mini-Shazam

This is a small fingerprint-matching demonstration using a library of
six **synthetic** songs and five-second queries. It is not a test
against a commercial music-recognition service or a large real-world
song collection.

  ------------------------------------------------------------------------------------------
  Query           Best match                           Score Runner-up score    Query hashes
  --------------- -------------------------- --------------- --------------- ---------------
  Clean music     `main_music (synthetic)`               363               1             392
  excerpt (sanity                                                            
  check)                                                                     

  ICA-separated   `main_music (synthetic)`               304               1             384
  music component                                                            

  Raw mixture     `main_music (synthetic)`                 6               2             555
  channel 1                                                                  
  ------------------------------------------------------------------------------------------

The clean excerpt and separated component matched the intended synthetic
track in this run. The fingerprint score is specific to this
implementation and library; it should not be compared directly with
scores from other systems.

## Files created by the run

  -----------------------------------------------------------------------
  Location                            Contents
  ----------------------------------- -----------------------------------
  `data/sources/`                     Original recordings and prepared 16
                                      kHz mono copies

  `data/mixtures/`                    `mixture_1.wav`, `mixture_2.wav`,
                                      `mixture_3.wav`

  `outputs/separated/`                `separated_speech.wav`,
                                      `separated_music.wav`,
                                      `separated_ambient.wav`, and
                                      `raw_components.npy`

  `outputs/plots/`                    Waveform, whitening, spectrogram,
                                      correlation, metric, convergence
                                      and bonus plots

  `outputs/metrics/`                  CSV/JSON metrics, mixing-matrix
                                      details, audio checks and
                                      `results_summary.md`
  -----------------------------------------------------------------------

The program checks that the generated WAV files have the expected sample
rate, contain finite non-silent audio, and that the plots are not empty.
The successful run reported **9 verified WAV files and 13 plots**.

## Running the tests

Run the unit tests from the project root:

``` powershell
python -m unittest discover -s tests -t . -v
```

Pytest can also be used if it is installed:

``` powershell
python -m pytest
```

## Notes and limitations

-   The main experiment is a controlled, instantaneous linear mixture.
    It does not model room reverberation, echoes, delays or changing
    mixtures.
-   ICA assumes the sources are sufficiently independent. Real audio may
    not satisfy this assumption perfectly.
-   The source recordings are required as references for the reported
    evaluation metrics. In a real blind-separation task, the original
    sources would usually not be available.
-   The reported numbers are from one run with seed `0`. Different
    recordings, seeds or settings can produce different results.
-   SI-SDR and correlation are related measures; they should not be
    treated as fully independent evidence of quality.
-   The mini-Shazam experiment uses a small synthetic library, so it
    only demonstrates the basic fingerprinting workflow.
-   The separated WAV files are scaled for convenient playback. Use the
    saved raw components and metrics when you need to inspect the
    algorithm's unscaled output.

## Troubleshooting

**FastICA does not converge:** Try another seed or adjust the FastICA
settings in `config.py`, such as the nonlinearity or maximum iteration
count.

**The covariance matrix is degenerate:** Check that the sources are not
identical or linearly dependent, and confirm that the mixing matrix has
full rank.

**A source file is rejected:** Make sure all three expected files exist,
contain actual audio, are long enough for the selected duration, and are
not silent or corrupted.

**The README results look old:** Run `python main.py` again. The program
updates the results section between the `RESULTS:START` and
`RESULTS:END` markers.

## Further work

Possible extensions include testing more recording sets, evaluating on
mixtures with reverberation, trying frequency-domain or convolutive ICA,
and comparing against established source-separation baselines.

## References

-   Hyvärinen, A., & Oja, E. (2000). Independent component analysis:
    Algorithms and applications. *Neural Networks, 13*(4--5), 411--430.
-   Wang, A. (2003). An industrial-strength audio search algorithm. In
    *Proceedings of the 4th International Conference on Music
    Information Retrieval (ISMIR)*.
