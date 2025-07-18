User Guide
==================================

SMRT (Snow Microwave Radiative Transfer Model) is a powerful and highly flexible framework designed to simulate the interaction of microwave radiation with natural ice-covered surfaces, including snow, sea-ice, and lake-ice. Written in Python, SMRT was originally developed with the European Space Agency's support to improve the representation of snow microstructure in microwave remote sensing. However, its modular and open-source nature has allowed it to expand into a more versatile tool for various ice and snow-related microwave remote sensing applications.
Key Features and Capabilities of SMRT
1. Modular and Extensible Architecture:
SMRT’s modular framework allows easy customization and extension. This "plug-and-play" design enables users to swap components easily for testing different models, assumptions, or configurations. Researchers can easily compare the results of different scattering models, microstructure representations, and solvers without needing to modify the entire codebase.
2. Advanced Scattering Models:
SMRT supports a variety of scattering models that account for the interactions between microwave radiation and the microstructure of snow, ice, and other surface materials. The following scattering models are available:
    • IBA (Integral Equation-Based Approximation): A model used to compute scattering and absorption properties, especially suitable for simulating the interactions of microwaves with layered or heterogeneous media for a variety of microstructure.
    • SCE (Strong Contrast Expansion): the cutting edge method to compute microwave scattering modeling for dense media. SCE addresses limitations in traditional scattering models that struggle with intermediate snow densities and high frequencies. It provides a more accurate way to calculate scattering coefficients by using a detailed microstructure model and the effective permittivity of heterogeneous, dielectric media.
    • DMRT (Dense Media Radiative Transfer): A fast model used to compute scattering and absorption properties of sticky or none sticky spheres, accounting for the high fractional volume of scatterers found in snow.
    • Rayleigh Independent Scattering: An approximation method for cases where the scattering is dominated by small, non-resonant particles (such as small ice crystals or snow grains).

3. Microstructure Representations:
The snow and ice microstructure—its composition, grain size, shape, and layering—heavily influences microwave scattering properties. SMRT includes a variety of microstructure representations, allowing researchers to simulate and analyze different physical characteristics of snow and ice:
    • Exponential Distribution: A probabilistic model to simulate random distributions of snow grain sizes or ice crystal shapes.
    • Sticky Hard Spheres: This model treats the snow or ice microstructure as hard spheres with sticky interactions, suitable for studying the effects of snow cohesion and aggregation.
    • Gaussian Random Fields: A model that represents snow and ice microstructure using spatially correlated random fields, making it ideal for studying more complex, realistic structures found in nature.
4. Radiative Transfer Solvers:
SMRT comes with several radiative transfer solver, which is used to compute the transmission, scattering, and absorption of microwaves through snow and ice:
- DORT (Discrete Ordinate Radiative Transfer): a general purpose solver for radiometers and radars that is robust and account for multiple scattering between the layers and the interfaces.
- First Order Iterative solution: a simple yet fast solver for radar applications that only compute single scattering but provide also the contribution of each scattering mechanisms.
- Nadir LRM Altimetry: a solver to compute altimetric waveform in the Low Rate Mode.
- Multi-Fresnel Thermal Emission: a very fast solver to compute thermal emission of non scattering layered media, such as the ice-sheet at low frequencies.

5. Snow, Sea-Ice, and Lake-Ice Applications:
SMRT is versatile enough to model not only snow but also other cryospheric materials such as sea-ice and lake-ice. Researchers can use it to study the effects of ice thickness, snow accumulation, temperature profiles, and other factors that influence microwave scattering and remote sensing measurements. This makes SMRT especially valuable for satellite-based remote sensing of the Earth's frozen regions, including studies related to climate change, ice sheet dynamics, and polar research.
6. Integration with Other Models:
SMRT offers wrappers that allow users to run other established models such as:
    • MEMLS (Microwave Emission Model for Layered Snowpacks): A model used to simulate microwave emission from layered snowpacks.
    • HUT (Helsinki University of Technology Snow Microwave Emission Model): Another widely-used snow microwave emission model.
    • DMRT-QMS: A model based on the Discrete-Ordinate Method, extended for complex snow and ice structures.
This ability to run multiple models within a single framework helps users validate and cross-check results, compare theoretical assumptions, and integrate different data sources.
7. Open-Source and Community-Driven:
One of the key strengths of SMRT is its open-source nature. As a community-driven model, SMRT encourages contributions from researchers, scientists, and engineers around the world. This collaborative approach ensures that the model continues to evolve, incorporating the latest scientific advancements, user feedback, and new methodologies. Whether you’re a beginner or an expert, the open-source design allows you to adapt the model to your specific research needs and contribute back to its development.
Looking Ahead: Future Developments
While SMRT provides a robust set of tools for current research, there are plenty of opportunities for future enhancements. Researchers are constantly working to integrate new theoretical advancements in radiative transfer modeling, improve solver efficiency, and extend SMRT’s capabilities to new materials and conditions. Whether it's developing new solvers, adding support for more complex microstructural models, or incorporating other types of remote sensing data, SMRT continues to evolve as a cutting-edge research tool.



Follow this user guide to learn how to install SMRT and to get started. A few tutorials then go through the most frequent use cases.

We then give recommendations for citing SMRT in publications that give results given by the model.

For more detailed documentation you can browse the API reference which gives the only continuously up-to-date reference for default behaviours as it is auto-generated from code source. For developers who want to implement new behaviour in SMRT for their own use or for improving SMRT, we recommend to read the :doc:`developer_guidelines` and to contact the authors of the model to discuss about the best/most generic approach to solve their problem.

.. toctree::
    :titlesonly:
    :maxdepth: 2

    Installation guide <install>
    Getting started <quick_start>
    Tutorials <tutorials/index>
    Cite SMRT <publish>