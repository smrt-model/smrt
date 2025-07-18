
Recommendations for publishing papers using SMRT
===================================================

SMRT differs from other models in the huge number of possible options that are readily available for users. The reader of your future paper, even if proficient in SMRT, may not understand exactly your results and the implications of your work unless all the selected options are precisely described. An obvious set of choices of options from someone is likely different for someone else.

This is especially critical when drawing positive or negative conclusions on SMRT performance to reproduce this or that result. Such statement is only relevant given the selected options. SMRT performance is very linked to the performance of the underlying models, theories and formulations that you have selected, and the combination of them. For this reason, credits (and complains) should refer to the used models, theories, or formulations rather than to SMRT in general. Only when the added-value of SMRT as a framework is involved in your conclusion (e.g. owing to its architecture, or a specific model only available in SMRT), “SMRT” can be directly credited (or blamed), using the reference to the precise paper that proposed the development (e.g. LRM altimetry was first presented in Larue et al. 2021) rather than the general SMRT v1.0 paper (Picard et al. 2018).

In practice, we would recommend the following:

- “SMRT” should be used to designate the modeling framework, not your specific results.

- SMRT-QCA, SMRT-IBA, SMRT-SCE should be used to designate the scattering theory family, unless working at low frequencies only. The exact module used can be indicated (in the method section for instance. e.g. symsce_torquato21).

- In the case of IBA and SCE, it is important to indicate the microstructure representation. «Exponential» can be understood as the default (as IBA was popularized in MEMLS that solely uses exponential representation). All other microstructure representations should be explicitly mentioned. It is also important to describe precisely how the grain size metrics is defined and used. The generic term “grain size” is insufficient. Options according to the microstructure representation are “sphere radius” as used in the sticky hard sphere model (Tsang et al. 1985), “correlation length” as described in Matzler, 2022, and “Microwave Grain Size” (MGS) as defined in Picard al. 2022. Examples of possible denomination: SMRT-IBA-EXP, SMRT-IBA-SHS.

- If rough interfaces are used, the scattering model should be indicated (GO, IEM, …). As several modules implement variants of these theories, it is better to indicate the exact module used.

- If a non-default permittivity formulation is used, it must be mentioned, with a reference to the original paper, which can be found in SMRT documentation, in principle.

- For sea-ice, the minimum information is whether “first-year ice” or “multi-year ice” has been used to represent the ice, since it drastically changes the scattering and absorption calculation. Any change of the permittivity (with respect to the default) should be mentioned as it has huge impact.

- The method to solve the radiative transfer method is not a critical information in general. The DORT solver has been the first and main solver for a long time in SMRT, it is understood as the default. If you use the altimetry modules, it is possible to refer to SMRT-Altim, with a reference to Larue et al. 2021 (for LRM) and Picard et al. 2025/26 (for SAR).

- In general, we recommend to indicate the version of SMRT (or the commit hash if the dev version is used) as some default options may change in the future. If you need a SMRT code doi from Zenodo (often required by the publishers) for your paper, don’t hesitate to contact us, we will provide it rapidly.

- We strongly recommend to publish the code running your simulations, at least the part configuring SMRT. This is the most detailed information that readers will need to understand your results.


All this information can also be summarized in a table, details to be adatped to the specific focus of your study.

+-----------------------------------------------+----------------------+
| Topic                                         | Description          |
+===============================================+======================+
| Scattering theory                             |                      |
+-----------------------------------------------+----------------------+
| Microstructure Representation                 |                      |
+-----------------------------------------------+----------------------+
| Surface/interface scattering models and       |                      |
| substrate                                     |                      |
+-----------------------------------------------+----------------------+
| Permittivity model snow                       |                      |
+-----------------------------------------------+----------------------+
| Sea ice type and permittivity model           |                      |
+-----------------------------------------------+----------------------+
| Atmosphere                                    |                      |
+-----------------------------------------------+----------------------+
| Radiative transfer Solver                     |                      |
+-----------------------------------------------+----------------------+
| SMRT version                                  |                      |
+-----------------------------------------------+----------------------+
