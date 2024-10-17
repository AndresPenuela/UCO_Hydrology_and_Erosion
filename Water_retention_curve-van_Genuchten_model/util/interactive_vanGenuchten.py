import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

def vanGenuchten(Ψ_m, α, n, *args):
    """
    Calculate effective saturation or water content based on the van Genuchten model.

    Parameters:
    Ψ_m : array-like
        Matric potential values.
    α : float
        Parameter related to the inverse of the air entry potential.
    n : float
        Parameter related to the pore size distribution.
    *args : optional
        If provided, the first two arguments are θ_r (residual water content) and θ_s (saturated water content).

    Returns:
    result : array
        Effective saturation (S_e) or water content (θ) depending on the presence of θ_r and θ_s.
    """
    m = 1 - 1/n  # Calculate the parameter m from n

    if args:  # Check if optional arguments (θ_r, θ_s) are provided
        θ_r = args[0]  # Residual water content
        θ_s = args[1]  # Saturated water content
        # Calculate water content θ using the van Genuchten equation
        θ = θ_r + (θ_s - θ_r) * (1 / (1 + (-α * -Ψ_m) ** n)) ** m
        result = θ  # Return calculated water content
    else:
        # Calculate effective saturation S_e using the van Genuchten equation
        S_e = (1 / (1 + (-α * -Ψ_m) ** n)) ** m
        result = S_e  # Return calculated effective saturation

    return result  # Return the result

def interactive_vanGenuchten_2(soil_type='sand'):
    """
    Simulate and plot the van Genuchten model for different soil types (sand, loam, clay).
    Each soil type has predefined van Genuchten parameters: α, n, θ_r (residual), θ_s (saturated).
    
    Parameters:
    soil_type : str
        The type of soil for which the model is simulated ('sand', 'loam', 'clay').
    """
    # Set van Genuchten parameters based on soil type
    if soil_type == 'sand':
        α = 0.0124; n = 6.66; θ_r = 0.01; θ_s = 0.26
    elif soil_type == 'loam':
        α = 0.0081; n = 2.1632; θ_r = 0.05; θ_s = 0.43
    elif soil_type == 'clay':
        α = 0.0066; n = 1.8601; θ_r = 0.20; θ_s = 0.47 

    # Observed matric potential (Ψ_m) and corresponding effective saturation (S_e) values
    Ψ_m_obs = np.array([20, 60, 300, 800, 5000, 10000])
    S_e_obs = np.array([0.951, 0.914, 0.432, 0.172, 0.042, 0.023])

    # Simulated range of matric potential values from 1 to 100,000
    Ψ_m_sim = np.arange(1, 100000 + 1, 1)
    
    # Convert observed effective saturation (S_e) to observed water content (θ_obs)
    θ_obs = S_e_obs * (θ_s - θ_r) + θ_r
    
    # Simulate water content (θ_sim) using the van Genuchten model
    θ_sim = vanGenuchten(Ψ_m_sim, α, n, θ_r, θ_s)
    # Convert simulated water content (θ_sim) to effective saturation (S_e_sim)
    S_e_sim = (θ_sim - θ_r) / (θ_s - θ_r)
    
    # Calculate field capacity and wilting point using specific matric potential values
    field_capacity = vanGenuchten(100, α, n, θ_r, θ_s)  # at Ψ_m = 100 cm
    wilting_point = vanGenuchten(15000, α, n, θ_r, θ_s)  # at Ψ_m = 15000 cm
    field_capacity_S_e = (field_capacity - θ_r) / (θ_s - θ_r)
    wilting_point_S_e = (wilting_point - θ_r) / (θ_s - θ_r)

    # Create a figure with 2 subplots
    plt.figure(figsize=(15, 4))  # Define the plot size
    
    ### Plot 1: Effective saturation (Se) vs Matric potential (Ψ)
    plt.subplot(1, 2, 1)  # First subplot
    plt.scatter(Ψ_m_obs, S_e_obs, label='measured')  # Plot observed data
    plt.plot(Ψ_m_sim, S_e_sim, label='simulated')  # Plot simulated data
    # Add field capacity and wilting point as vertical and horizontal lines
    plt.vlines(100, 0, field_capacity_S_e, linestyle='--', color='g', label='field capacity')
    plt.vlines(15000, 0, wilting_point_S_e, linestyle='--', color='r', label='wilting point')
    plt.hlines(field_capacity_S_e, 0, 100, linestyle='--', color='g')
    plt.hlines(wilting_point_S_e, 0, 15000, linestyle='--', color='r')
    plt.ylabel('Se [vol/vol]')  # Y-axis label
    plt.xlabel('Ψm [|cm|]')  # X-axis label
    plt.xscale("log")  # Set X-axis to logarithmic scale
    plt.title('Effective saturation (Se) vs Matric potential (Ψ)')  # Title
    plt.legend()  # Show legend
    
    ### Plot 2: Soil water content (θ) vs Matric potential (Ψ)
    plt.subplot(1, 2, 2)  # Second subplot
    plt.scatter(Ψ_m_obs, θ_obs, label='measured')  # Plot observed data
    plt.plot(Ψ_m_sim, θ_sim, label='simulated')  # Plot simulated data
    # Add field capacity and wilting point as vertical and horizontal lines
    plt.vlines(100, 0, field_capacity, linestyle='--', color='g', label='field capacity')
    plt.vlines(15000, 0, wilting_point, linestyle='--', color='r', label='wilting point')
    plt.hlines(field_capacity, 0, 100, linestyle='--', color='g')
    plt.hlines(wilting_point, 0, 15000, linestyle='--', color='r')
    plt.ylabel('θ [vol/vol]')  # Y-axis label
    plt.ylim([0, 1])  # Limit Y-axis from 0 to 1
    plt.xlabel('Ψm [|cm|]')  # X-axis label
    plt.xscale("log")  # Set X-axis to logarithmic scale
    plt.title('Soil water content (θ) vs Matric potential (Ψ)')  # Title
    plt.legend()  # Show legend
    plt.show()  # Display the plot

def interactive_vanGenuchten_3(p1=0.004, p2=1.2, p3=0.05, p4=0.3):
    """
    Simulate and plot the van Genuchten model with custom parameters and calculate available water.
    
    Parameters:
    p1 : float
        Parameter α (inverse air entry potential).
    p2 : float
        Parameter n (related to pore size distribution).
    p3 : float
        Residual water content (θ_r).
    p4 : float
        Saturated water content (θ_s).
    """
    # Simulated range of matric potential values from 1 to 100,000
    Ψ_m_sim = np.arange(1, 100000 + 1, 1)
    
    # Simulate water content (θ_sim) using the van Genuchten model
    θ_sim = vanGenuchten(Ψ_m_sim, p1, p2, p3, p4)
    # Calculate field capacity and wilting point
    field_capacity = vanGenuchten(100, p1, p2, p3, p4)  # at Ψ_m = 100 cm
    wilting_point = vanGenuchten(15000, p1, p2, p3, p4)  # at Ψ_m = 15000 cm
    available_water = field_capacity - wilting_point  # Calculate available water (field capacity - wilting point)

    ### Plot of Soil water content (θ) vs Matric potential (Ψ)
    plt.figure(figsize=(6, 4))  # Define the plot size
    plt.plot(Ψ_m_sim, θ_sim)  # Plot simulated water content
    # Add field capacity and wilting point as vertical and horizontal lines
    plt.vlines(100, 0, field_capacity, linestyle='--', color='g', label='field capacity')
    plt.vlines(15000, 0, wilting_point, linestyle='--', color='r', label='wilting point')
    plt.hlines(field_capacity, 0, 100, linestyle='--', color='g')
    plt.hlines(wilting_point, 0, 15000, linestyle='--', color='r')
    plt.ylabel('θ [vol/vol]')  # Y-axis label
    plt.ylim([0, 1])  # Limit Y-axis from 0 to 1
    plt.xlabel('Ψm [|cm|]')  # X-axis label
    plt.xscale("log")  # Set X-axis to logarithmic scale
    plt.title('Available water (field capacity - wilting point) = %.3f' % available_water)  # Title
    plt.legend()  # Show legend
    plt.show()  # Display the plot
