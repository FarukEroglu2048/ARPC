import numpy
import scipy.optimize

import matplotlib.pyplot

earth_gravitational_acceleration = 9.80665 #m*s^-2
universal_gas_constant = 8.31446261815324 #J*K^-1*mol^-1
avogadro_constant = 6.02214076e23 #mol^-1
air_molar_mass = 0.02897 #kg*mol^-1
air_refractive_index = 1.000293 #Unitless

msl_temperature = 288.15 #K
msl_pressure = 101325.0 #Pa

red_wavelength = 6.11e-7 #m
green_wavelength = 5.49e-7 #m
blue_wavelength = 4.64e-7 #m

atmosphere_layer_altitudes, atmosphere_layer_temperature_gradients = numpy.genfromtxt("atmosphere_layers.txt", missing_values=(None, "End"), unpack=True, loose=False)
atmosphere_layer_temperature_gradients = atmosphere_layer_temperature_gradients[:-1]

atmosphere_layer_altitudes *= 1000.0 #m
atmosphere_layer_temperature_gradients *= 0.001 #K*m^-1

ozone_altitudes, ozone_number_densities = numpy.genfromtxt("ozone_number_densities.txt", unpack=True, loose=False)

ozone_altitudes *= 1000.0 #m
ozone_number_densities *= 1000000.0 #molecules*m^-3

ozone_wavelengths, ozone_absorption_cross_sections = numpy.genfromtxt("ozone_absorption_cross_sections.txt", unpack=True, loose=False)

ozone_wavelengths *= 1.0e-9 #m
ozone_absorption_cross_sections *= 0.0001 #m^2*molecules^-1

cie_1931_wavelengths, cie_1931_x, cie_1931_y, cie_1931_z = numpy.genfromtxt("cie_1931_standard_observer.txt", unpack=True, loose=False)

cie_1931_wavelengths *= 1.0e-9 #m
cie_1931_xyz = numpy.stack((cie_1931_x, cie_1931_y, cie_1931_z), axis=1)

"""
Calculates the air temperature for a layered atmosphere model with constant temperature gradients within a layer, such as ISA.

See page 10 in U.S. Standard Atmosphere, 1976 for the equation used in this implementation.
"""
def air_temperature(altitude):
    temperature = msl_temperature

    for current_layer_index in range(atmosphere_layer_altitudes.size - 1):
        current_layer_altitude = atmosphere_layer_altitudes[current_layer_index]
        next_layer_altitude = atmosphere_layer_altitudes[current_layer_index + 1]
        
        current_temperature_gradient = atmosphere_layer_temperature_gradients[current_layer_index]

        if altitude < next_layer_altitude:
            temperature += current_temperature_gradient * (altitude - current_layer_altitude)

            break
        else:
            temperature += current_temperature_gradient * (next_layer_altitude - current_layer_altitude)

    return temperature

"""
Calculates the air pressure for a layered atmosphere model with constant temperature gradients within a layer, such as ISA.

See page 12 in U.S. Standard Atmosphere, 1976 for the equations used in this implementation.
"""
def air_pressure(altitude):
    pressure = msl_pressure

    for current_layer_index in range(atmosphere_layer_altitudes.size - 1):
        current_layer_altitude = atmosphere_layer_altitudes[current_layer_index]
        next_layer_altitude = atmosphere_layer_altitudes[current_layer_index + 1]

        current_temperature_gradient = atmosphere_layer_temperature_gradients[current_layer_index]

        if altitude < next_layer_altitude:
            if current_temperature_gradient == 0.0:
                pressure *= numpy.exp((-1.0 * earth_gravitational_acceleration * air_molar_mass * (altitude - current_layer_altitude)) / (universal_gas_constant * air_temperature(current_layer_altitude)))
            else:
                pressure *= numpy.power(air_temperature(current_layer_altitude) / air_temperature(altitude), (earth_gravitational_acceleration * air_molar_mass) / (universal_gas_constant * current_temperature_gradient))

            break
        else:
            if current_temperature_gradient == 0.0:
                pressure *= numpy.exp((-1.0 * earth_gravitational_acceleration * air_molar_mass * (next_layer_altitude - current_layer_altitude)) / (universal_gas_constant * air_temperature(current_layer_altitude)))
            else:
                pressure *= numpy.power(air_temperature(current_layer_altitude) / air_temperature(next_layer_altitude), (earth_gravitational_acceleration * air_molar_mass) / (universal_gas_constant * current_temperature_gradient))

    return pressure

"""
Calculates the number density of air.

See page 15 in U.S. Standard Atmosphere, 1976 for the equation used in this implementation.
"""
def air_number_density(altitude):
    return (avogadro_constant * air_pressure(altitude)) / (universal_gas_constant * air_temperature(altitude))

"""
Fittable curve for the air number density.
"""
def air_number_density_ratio(altitude, scale_height):
    return numpy.exp(-1.0 * (altitude / scale_height))

"""
Fittable curve for the ozone number density.
"""
def ozone_number_density_ratio(altitude, layer_base, layer_thickness):
    return numpy.maximum(1.0 - (numpy.abs(altitude - layer_base) / (layer_thickness / 2.0)), 0.0)

"""
Calculates the rayleigh scattering coefficient for air.

See page 3 in Display of The Earth Taking into Account Atmospheric Scattering, 1993 from Nishita et al. for the equation used in this implementation.
"""
def air_rayleigh_scattering_coefficient(wavelength):
    return (8.0 * numpy.power(numpy.pi, 3.0) * numpy.power(numpy.power(air_refractive_index, 2.0) - 1.0, 2.0)) / (3.0 * air_number_density(0.0) * numpy.power(wavelength, 4.0))

"""
Calculates the absorption coefficient for ozone.
"""
def ozone_absorption_coefficient(wavelength):
    return numpy.interp(wavelength, ozone_wavelengths, ozone_absorption_cross_sections) * ozone_number_densities.max()

"""
Converts scattering and absorption coefficients for the entire visual spectrum into a color component representation
"""
def coefficients_color_component_representation(coefficients, color_matching_function):
    return numpy.maximum(numpy.sum(coefficients * color_matching_function) / numpy.sum(color_matching_function), 0.0)

print("\n######## Atmosphere Rendering Parameter Calculator ########")

msl_temperature_string = input("\nEnter the air temperature at mean sea level in Celsius, leave empty to use the standard mean sea level temperature of U.S. Standard Atmosphere, 1976: ")
msl_pressure_string = input("Enter the air pressure at mean sea level in millibars, leave empty to use the standard mean sea level pressure of U.S. Standard Atmosphere, 1976: ")

if (msl_temperature_string != ""):
    msl_temperature = float(msl_temperature_string) + 273.15

if (msl_pressure_string != ""):
    msl_pressure = float(msl_pressure_string) * 100.0

air_altitudes = numpy.linspace(0.0, atmosphere_layer_altitudes[-1], 10000)
air_number_densities = numpy.empty_like(air_altitudes)

for index, altitude in numpy.ndenumerate(air_altitudes):
    air_number_densities[index] = air_number_density(altitude)

air_number_density_parameters, air_number_density_covariances = scipy.optimize.curve_fit(air_number_density_ratio, air_altitudes, air_number_densities / air_number_density(0.0), bounds=(0.0, atmosphere_layer_altitudes[-1]))
air_scale_height = air_number_density_parameters[0]

ozone_number_density_parameters, ozone_number_density_covariances = scipy.optimize.curve_fit(ozone_number_density_ratio, ozone_altitudes, ozone_number_densities / ozone_number_densities.max(), bounds=(0.0, ozone_altitudes[-1]))

ozone_base = ozone_number_density_parameters[0]
ozone_thickness = ozone_number_density_parameters[1]

air_rayleigh_scattering_coefficients = numpy.empty_like(cie_1931_wavelengths)
ozone_absorption_coefficients = numpy.empty_like(cie_1931_wavelengths)

for index, wavelength in numpy.ndenumerate(cie_1931_wavelengths):
    air_rayleigh_scattering_coefficients[index] = air_rayleigh_scattering_coefficient(wavelength)
    ozone_absorption_coefficients[index] = ozone_absorption_coefficient(wavelength)

srgb_red = numpy.dot(cie_1931_xyz, numpy.array((3.2406, -1.5372, -0.4986), dtype=float))
srgb_green = numpy.dot(cie_1931_xyz, numpy.array((-0.9689, 1.8758, 0.0415), dtype=float))
srgb_blue = numpy.dot(cie_1931_xyz, numpy.array((0.0557, -0.204, 1.057), dtype=float))

print("\n#### Parameters for Air ####")

standard_notation_format_string = "{0:.2f}"
scientific_notation_format_string = "{0:.4e}"

print("\nScale Height: " + standard_notation_format_string.format(air_scale_height) + " m")

print("\nRayleigh Scattering Coefficients for Air:")
print("Red: " + scientific_notation_format_string.format(coefficients_color_component_representation(air_rayleigh_scattering_coefficients, srgb_red)) + " m^-1, Green: " + scientific_notation_format_string.format(coefficients_color_component_representation(air_rayleigh_scattering_coefficients, srgb_green)) + " m^-1, Blue: " + scientific_notation_format_string.format(coefficients_color_component_representation(air_rayleigh_scattering_coefficients, srgb_blue)) + " m^-1")

print("\n#### Parameters for Ozone ####")

print("\nLayer Base: " + standard_notation_format_string.format(ozone_base) + " m, Layer Thickness: " + standard_notation_format_string.format(ozone_thickness) + " m")

print("\nAbsorption Coefficients for Ozone:")
print("Red: " + scientific_notation_format_string.format(coefficients_color_component_representation(ozone_absorption_coefficients, srgb_red)) + " m^-1, Green: " + scientific_notation_format_string.format(coefficients_color_component_representation(ozone_absorption_coefficients, srgb_green)) + " m^-1, Blue: " + scientific_notation_format_string.format(coefficients_color_component_representation(ozone_absorption_coefficients, srgb_blue)) + " m^-1")

while True:
    input_string = input("\nDo you want to see the visualizations of the atmosphere? Type \"Yes\" to continue with the visualizations, type \"No\" or leave empty to exit without the visualizations: ")

    if (input_string == "Yes"):
        air_figure, air_subplots = matplotlib.pyplot.subplots(ncols=3)

        air_subplots[0].plot(air_number_densities, air_altitudes, label="Calculated")
        air_subplots[0].plot(air_number_density_ratio(air_altitudes, air_scale_height) * air_number_density(0.0), air_altitudes, label="Fitted")

        air_subplots[0].set_title("Air Number Density vs Altitude")

        air_subplots[0].set_xlabel("Number Density (molecules*m^-3)")
        air_subplots[0].set_ylabel("Altitude (m)")

        air_subplots[0].grid(visible=True, linestyle="dashed")
        air_subplots[0].legend()

        air_temperatures = numpy.empty_like(air_altitudes)
        air_pressures = numpy.empty_like(air_altitudes)

        for index, altitude in numpy.ndenumerate(air_altitudes):
            air_temperatures[index] = air_temperature(altitude)
            air_pressures[index] = air_pressure(altitude)

        air_subplots[1].plot(air_temperatures, air_altitudes, label="Calculated")

        air_subplots[1].set_title("Air Temperature vs Altitude")

        air_subplots[1].set_xlabel("Temperature (K)")
        air_subplots[1].set_ylabel("Altitude (m)")

        air_subplots[1].grid(visible=True, linestyle="dashed")
        air_subplots[1].legend()

        air_subplots[2].plot(air_pressures, air_altitudes, label="Calculated")

        air_subplots[2].set_title("Air Pressure vs Altitude")

        air_subplots[2].set_xlabel("Pressure (Pa)")
        air_subplots[2].set_ylabel("Altitude (m)")

        air_subplots[2].grid(visible=True, linestyle="dashed")
        air_subplots[2].legend()

        ozone_figure, ozone_subplot = matplotlib.pyplot.subplots()

        ozone_subplot.plot(ozone_number_densities, ozone_altitudes, label="Measured")
        ozone_subplot.plot(ozone_number_density_ratio(ozone_altitudes, ozone_base, ozone_thickness) * ozone_number_densities.max(), ozone_altitudes, label="Fitted")

        ozone_subplot.set_title("Ozone Number Density vs Altitude")

        ozone_subplot.set_xlabel("Number Density (molecules*m^-3)")
        ozone_subplot.set_ylabel("Altitude (m)")

        ozone_subplot.grid(visible=True, linestyle="dashed")
        ozone_subplot.legend()

        matplotlib.pyplot.show()

        break
    elif (input_string == "No") or (input_string == ""):
        break
    else:
        print("\nInvalid input!")