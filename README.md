# Multi-Sensor State and Uncertainty Estimation

This project provides a modular, auto-differentiable, and parallelizable codebase for estimating the state and uncertainty of a system using multiple sensors. It also includes implementations of multiple sensor, dynamics, and filter modules to provide flexibility and adaptability to different use cases. The project emphasizes ease of extendibility and tools for dedicated uncertainty characterization.

## Features

- Modular architecture for easy extension and customization
- Auto-differentiation for efficient computation of gradients
- Parallelizable implementation for faster performance
- Multiple sensor, dynamics, and filter modules included
- Dedicated uncertainty characterization for accurate state estimation

## Usage

To use the codebase, clone the repository and install the necessary dependencies:

```
git clone https://github.com/Stanford-NavLab/DDUncertaintyFilter.git
cd DDUncertaintyFilter
pip install -r requirements.txt
```

Then, import the necessary modules and start using the provided functions to estimate the state and uncertainty of your system. See the `notebooks/` directory for sample code that demonstrates how to use the codebase.

## Contributing

Contributions are welcome! To contribute, fork the repository and create a new branch for your changes. Once you've made your changes, submit a pull request and we'll review it as soon as possible.

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

