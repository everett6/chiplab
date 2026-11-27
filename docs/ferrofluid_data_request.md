# Draft: data request to the authors (not sent)

Before sending: confirm the corresponding author's address on the paper itself (the preprint
lists alessandro.chiolerio@iit.it; it reached us through a summary of the web page), and
fill in the bracketed fields.

---

**To:** corresponding author, Dr. Alessandro Chiolerio

**Subject:** Data request: in-memory computing in a ferrofluid (Adv. Mater. 2023, 10.1002/adma.202211406)

Dear Dr. Chiolerio,

I am [name, affiliation or "an independent developer"]. I am building a simulator and would
like to test it against your two-port Fe3O4 ferrofluid experiment. I have read the main text
of the preprint (arXiv:2211.08152v1), which says the data are available on request. My model
is empirical and has not yet been compared with any measurement.

Would you be willing to share the following?

**Most important**

1. The voltage sweep of Fig. 1C (±3.8 V, 0.1 V per 1 s, 50 repeats).
2. The pulse-duration memory test of Fig. 2, including the reset phase and hold phase.
3. The multi-day run showing the shrinking impedance dynamics (supplement S7), and any
   record of the recovery.

For each: impedance or S-parameters against time (ideally Touchstone `.s2p`, version 1, one
file per sweep with its time stamp), and the voltage applied to each port against time
(CSV is ideal). If possible, **two independent runs** of each, so that I can fit on one and
test on the other.

**Needed to interpret them**

4. Which port or ports receive the programming voltage, how the other is terminated, the
   order of the voltage sweep, and the reset loop's voltages, timing and target.
5. VNA model, frequency grid, RF power, IF bandwidth, sweep time, calibration reference
   plane and bias-tee models.
6. The exact definition of the summed impedance (magnitude, real part or complex sum).
7. Repeat sweeps of an undisturbed sample, or your estimate of measurement uncertainty.
8. Sample composition, particle size, concentration and volume; electrode and vessel
   geometry; temperature during the runs.
9. Whether you have an explanation for Z12 differing from Z21 (2320 and 2060 Ω at the start
   of Fig. 1C), for example bias, calibration or drift between sweeps.

A partial set would still be useful. Please also tell me under what terms I may use the data,
whether any of it may be redistributed, and how you would like it cited. I am happy to send
back the comparison, including where the model fails.

Thank you for your time,

[name]
[contact details]
