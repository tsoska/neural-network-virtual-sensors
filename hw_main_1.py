# Loading Libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import textwrap
from sklearn.metrics import r2_score
import wandb
import csv
import shap

# Για επαναληψιμότητα των αποτελεσμάτων
np.random.seed(42)
torch.manual_seed(42)

wandb.login()

with open("results.csv", "w", newline="") as f: # For WandB results renew
    pass

sns.set_style("whitegrid") # για καλύτερη εποπτεία

# Loading Dataset
# file_path = r"C:\Users\georg\Downloads\Advanced_Control_HW_1\Data2026.csv"
folder_path = os.path.dirname(os.path.abspath(__file__)) #path to save or open files
file_path = os.path.join(folder_path, "Data2026.csv")
df = pd.read_csv(file_path, sep="\t", skiprows=[1]) # γιατί είναι tab-separated και το πρώτο row είναι μονάδες
df=df.drop(columns=["Unnamed: 11"]) # γιατι έβγαζε και μια στήλη με NaN στο τέλος


# Defining the fonts before plotting:
plt.rcParams.update({
    'font.family': 'sans-serif',  # monospace font
    'font.sans-serif': 'Calibri',  # specific sans-serif font
    'font.size': 20,
    'axes.titlesize': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 20
}) 

# πρωταρχικός έλεγχος των δεδομένων
print(df.head()) # ενδεικτικά πρώτες γραμμές 
print(df.shape) # πλήθος rows και columns
print(df.columns) # ονόματα στηλών
print(df.info()) # data types
print(df.describe().T) # βασικά στατιστικά

print("\n #################### Question 1 ########################## \n")
# Ερώτημα 1 - Μερικά διαγράμματα για ανάλυση του dataset 
df_no_time = df.drop(columns=["Time"]) # ο χρόνος δεν έχει νόημα, είναι ένα index

def heatmap(df, title):
    corr_matrix = df.corr()
    corr_matrix = corr_matrix.iloc[:-1, 1:]  # αφαίρεσε την πρώτη στήλη

    fig, ax = plt.subplots()

    n_rows, n_cols = corr_matrix.shape
    # Φτιάξε το mask με το σωστό shape (ορθογώνιο)
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for i in range(n_rows):
        for j in range(n_cols):
            if j <= i - 1:  # κάτω τρίγωνο (μετατοπισμένο κατά 1 λόγω crop)
                mask[i, j] = True

    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f",
                linewidths=0.5, linecolor='lightgray', cbar=True, ax=ax)
    
    # Κάλυψε τα masked κελιά με λευκά patches
    for i in range(n_rows):
        for j in range(n_cols):
            if mask[i, j]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                           facecolor='white', edgecolor='white', lw=1, zorder=3))
    ax.set_facecolor('white')
    plt.title(title)
    
    x_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    y_labels = [tick.get_text() for tick in ax.get_yticklabels()]
    wrapped_labels_x = [textwrap.fill(str(l).replace('_', ' '), width=5, break_long_words=False) for l in x_labels]
    wrapped_labels_y = [textwrap.fill(str(l).replace('_', ' '), width=15, break_long_words=False) for l in y_labels]
    ax.set_xticklabels(wrapped_labels_x, rotation=0, ha='center', multialignment='center')
    ax.set_yticklabels(wrapped_labels_y, rotation=0, ha='center', va='center', multialignment='center')
    ax.tick_params(axis='y', which='major', pad=70)
    plt.subplots_adjust(left=0.13, right=1.07, top=0.96, bottom=0.15)

    targets = ['NOx', 'lambda', 'Fuel Consumption', 'Intake Pressure']
    from matplotlib.patches import Rectangle
    for target in targets:
        if target in corr_matrix.index:
            row_idx = list(corr_matrix.index).index(target)
            col_start = row_idx      # ξεκινά από το πρώτο έγχρωμο κελί
            col_width = n_cols - col_start  # μέχρι το τέλος
            ax.add_patch(Rectangle((col_start, row_idx), col_width, 1,
                                    fill=False, edgecolor='black', lw=2.5, 
                                    clip_on=False, zorder=4))
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    plot=plt.show()
    return plot

# heatmap(df_no_time, "Initial Correlation Matrix")

# scatter plots for high correlation pairs
scatter_pairs = [
    # ("NOx", "Rot. Speed"),
    # ("NOx", "Engine Torque"),
    # ("Fuel Consumption", "Rot. Speed"),
    # ("Fuel Consumption", "Engine Torque"),

    # ("Fuel Consumption", "lambda"),
    # ("Fuel Consumption", "Exhaust Gas Mass Flow"),
    # ("Fuel Consumption", "Intake Pressure"),
    # ("Fuel Consumption", "Engine Torque"),
    # ("Fuel Consumption", "EGR Command"),
    # ("Fuel Consumption", "Exhaust Gas Temperature"),

    # ("lambda", "Engine Torque"),
    # ("lambda", "EGR Command"),

    # ("Exhaust Gas Mass Flow", "Intake Pressure"),
    # ("Exhaust Gas Mass Flow", "Rot. Speed"),
    # ("Exhaust Gas Mass Flow", "Engine Torque"),
    # ("Exhaust Gas Mass Flow", "EGR Command"),

    # ("Intake Pressure", "Rot. Speed"),
    # ("Intake Pressure", "Engine Torque"),
    # ("Intake Pressure", "EGR Command"),
    # ("Intake Pressure", "Exhaust Gas Temperature"),

    # ("Engine Torque", "EGR Command"),
    # ("Engine Torque", "Exhaust Gas Temperature"),

    # ("EGR Command", "Exhaust Gas Temperature"),
]

for x_col, y_col in scatter_pairs:
    fig,ax=plt.subplots()
    ax.scatter(df[x_col], df[y_col], color='#9671bd', s = 60, 
               edgecolors = '#6a408d',
               linewidths = 1.5,
               zorder = 3)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25) # transparent major grid
    ax.minorticks_on()
    ax.grid(True,which='minor', linestyle='-', linewidth=0.25, alpha=0.15) # transparent minor grid
    ax.set_axisbelow(True) # to get grid below data
    plt.tight_layout()
    plt.subplots_adjust(left=0.07, right=0.98, top=0.96, bottom=0.1)
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')
    plt.show()
    
print("\n #################### Question 2 ########################## \n")
# Ερώτημα 2 - Φιλτράρισμα δεδομένων
df = df[(df.drop(columns=["Time"]) > 0).all(axis=1)] # όλα τα μεγέθη πρέπει να έχουν θετικές τιμές
df = df.drop_duplicates() # remove dublicates
df = df[df["Torque Reference"] <= 100] # γιατί είναι ποσοστό
df = df[df["EGR Command"] <= 100] # γιατί είναι ποσοστό

# Remove everything above 200% from the mean value (obviously all data>0 because of filters)
def remove_outliers(df, column):
    mean = df[column].mean()
    upper_threshold = 3*mean
    return df[(df[column] <= upper_threshold)]

for col in df.columns:
    if col != "Time":
        df = remove_outliers(df, col)

df = df.drop(columns=["Time"])
print(df.shape) # για να δούμε πόσα δεδομένα έμειναν μετά το φιλτράρισμα

file_path_data=os.path.join(folder_path, "data.csv")
df.to_csv(file_path_data, sep='\t', index=False)

# heatmap(df, "Filtered Correlation Matrix")

# γκρουπάρισμα με βάση τα rpm, γιατι στα δεδομένα παρατηρούνται clusters
df["rpm_group"] = (df["Rot. Speed"] / 100).round().astype(int) * 100
#df.sort_values("rpm_group", inplace=True) # sort by rpm group
print(df["rpm_group"].value_counts().sort_index())

file_path_sorted=os.path.join(folder_path, "data_sorted.csv")
df.to_csv(file_path_sorted, sep='\t', index=False)

print("\n #################### Question 3 ########################## \n")
# Ερώτημα 3 - Split των δεδομένων σε training, validation και test set
# input data και target πρόβλεψης
X = df.drop(columns=["NOx", "Fuel Consumption", "Intake Pressure", "lambda", "rpm_group", "Torque Reference"]) # αφαιρούμε τα μεγέθη που δεν έχουν νόημα για την πρόβλεψη των NOx (time) ή είναι πολύ άμεσα συνδεδεμένα με αυτά
X_simple = X.copy()
feature_set_name = "Simple"
print(X.columns)
# αφαιρούμε τα μεγέθη που δεν έχουν νόημα για την πρόβλεψη των NOx (time) ή είναι πολύ άμεσα συνδεδεμένα με αυτά
y = df["NOx"]

# Σχεδίαση του νευρωνικού με κλάση 
class NNModel(nn.Module):
    def __init__(self, input_size, neurons_number, output_size):
        super(NNModel, self).__init__()
        layers=[]
        in_size=input_size
        for neurons in neurons_number:
            layers.append(nn.Linear(in_size, neurons)) # from the first layer to the next
            layers.append(nn.ReLU()) # non linearity
            in_size = neurons
        layers.append(nn.Linear(in_size, output_size)) # from the last hidden layer to the output
        
        self.net = nn.Sequential(*layers)
       
    def forward(self, x):
        return self.net(x) # εκτέλεση της forward διάδοσης δεδομένων στο νευρωνικό

# Ορισμός συνάρτησης κόστους και optimizer
loss_function = nn.MSELoss() # γιατί είναι πρόβλεψη συνεχούς μεγέθους, οπότε τιμωρούμε παραπάνω τις πιο απομακρυσμένες προβλέψεις

# Weights And Biases
# For simple models
sweep_config_simple = {
    "method": "grid",
    "metric": {"name": "Test_RMSE", "goal": "minimize"},
    "parameters": {
        "neurons": {"values": [[16], [20]]}, # set hidden layers and neurons for compare (1 hidden with 20 neurons max)
        "lr":      {"values": [0.001, 0.01]}, # set learnign rates for compare
        "epochs":  {"values": [500, 1000]} # set epochs for compare
    }
}
# For complex models
sweep_config_complex = {
    "method": "grid",
    "metric": {"name": "Test_RMSE", "goal": "minimize"},
    "parameters": {
        "neurons": {"values": [[32], [32, 32], [64, 64]]},
        "lr":      {"values": [0.001, 0.01]},
        "epochs":  {"values": [200, 500]}
    }
}
# Συνάρτηση που καλείται από το WandB sweep για κάθε διαφορετικό configuration
def train_Fuel_Consumption(model):
    wandb.init(project="FOC_nn")
    config = wandb.config # ανάγνωση των hyperparameters του τρέχοντος sweep run
    wandb.run.name = f"{feature_set_name}_Fuel_Consumption_neurons{config.neurons}_lr{config.lr}_epochs{config.epochs}"
    run_neural_network(
        X,
        df["Fuel Consumption"],
        model,
        config.epochs,
        config.neurons,
        config.lr,
        "Actual Fuel Consumption (scaled)",
        "Predicted Fuel Consumption (scaled)",
        "Actual Fuel Consumption [kg/h]",
        "Predicted Fuel Consumption [kg/h]",
        "Training vs Validation Loss for Fuel Consumption Model",
        "Predicted vs Actual Fuel Consumption",
        "Predicted vs Actual Fuel Consumption"
    )

def train_NOx(model):
    wandb.init(project="NOx_nn")
    config = wandb.config
    wandb.run.name = f"{feature_set_name}_NOx_neurons{config.neurons}_lr{config.lr}_epochs{config.epochs}"
    run_neural_network(
        X,
        df["NOx"],
        model,
        config.epochs, 
        config.neurons, 
        config.lr,
        "Actual NOx (scaled)",
        "Predicted NOx (scaled)",
        "Actual NOx [ppm]",
        "Predicted NOx [ppm]",
        "Training vs Validation Loss for NOx Model",
        "Predicted vs Actual NOx",
        "Predicted vs Actual NOx"
    )

def train_Intake_Pressure(model):
    wandb.init(project="Intake_Pressure_nn")
    config = wandb.config
    wandb.run.name = f"{feature_set_name}_Intake_Pressure_neurons{config.neurons}_lr{config.lr}_epochs{config.epochs}"
    run_neural_network(
        X,
        df["Intake Pressure"],
        model,
        config.epochs, 
        config.neurons, 
        config.lr,
        "Actual Intake Pressure (scaled)",
        "Predicted Intake Pressure (scaled)",
        "Actual Intake Pressure [kPa]",
        "Predicted Intake Pressure [kPa]",
        "Training vs Validation Loss for Intake Pressure Model",
        "Predicted vs Actual Intake Pressure",
        "Predicted vs Actual Intake Pressure"
    )

def train_lambda(model):
    wandb.init(project="lambda_nn")
    config = wandb.config
    wandb.run.name = f"{feature_set_name}_lambda_neurons{config.neurons}_lr{config.lr}_epochs{config.epochs}"
    run_neural_network(
        X,
        df["lambda"],
        model,
        config.epochs, 
        config.neurons, 
        config.lr,
        "Actual lambda (scaled)",
        "Predicted lambda (scaled)",
        "Actual lambda",
        "Predicted lambda",
        "Training vs Validation Loss for lambda Model",
        "Predicted vs Actual lambda",
        "Predicted vs Actual lambda"
    )

# Training
def run_neural_network(X, y, model_name, epochs, neurons_number, learning_rate, scaled_xlabel, scaled_ylabel, real_xlabel, real_ylabel, plot_title_loss, plot_title_scaled, plot_title_real, do_shap=False):
    np.random.seed(42)
    torch.manual_seed(42)
    print("\n #################### " + model_name + " Model ########################## \n")
    # training set (70%), validation set (10%) and test set (20%)
    rpm_groups = df["rpm_group"].unique()
    # Split ανά ομάδα στροφών
    train_groups, rest_groups = train_test_split(
    rpm_groups,
    test_size=0.3,
    random_state=42
    )

    val_groups, test_groups = train_test_split(
    rest_groups,
    test_size=20/30,
    random_state=42
    )
    # Έλεγχος των ομάδων rpm του κάθε set
    print("Train RPM groups:", sorted(train_groups))
    print("Validation RPM groups:", sorted(val_groups))
    print("Test RPM groups:", sorted(test_groups))

    # masks για να επιλεχθούν όλα τα data rows της κάθε ομάδας στροφών
    train_mask = df["rpm_group"].isin(train_groups)
    val_mask = df["rpm_group"].isin(val_groups)
    test_mask = df["rpm_group"].isin(test_groups)

    # Εφαρμογή των masks για τα τελικά sets
    X_train = X.loc[train_mask]
    X_val = X.loc[val_mask]
    X_test = X.loc[test_mask]
    y_train = y.loc[train_mask]
    y_val = y.loc[val_mask]
    y_test = y.loc[test_mask]

    # Κανονικοποίηση των δεδομένων
    x_scaler = StandardScaler() # κανονικόποιση ως z = (x - μ) / σ, γιατί δουλεύει καλά με ReLU
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train)   # scale στο training set
    X_val   = x_scaler.transform(X_val)         # scale στο validation
    X_test  = x_scaler.transform(X_test)        # scale στο test

    y_train = y_scaler.fit_transform(y_train.to_numpy().reshape(-1,1)) # reshape σε (N,1) για να ταιριάζει με το output του μοντέλου
    y_val   = y_scaler.transform(y_val.to_numpy().reshape(-1,1)) # το to_numpy() για να είναι συμβατό με το torch.tensor
    y_test  = y_scaler.transform(y_test.to_numpy().reshape(-1,1))

    # Μετατροπή σε tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32) 
    y_train_t = torch.tensor(y_train, dtype=torch.float32) 

    X_val_t = torch.tensor(X_val, dtype=torch.float32) 
    y_val_t = torch.tensor(y_val, dtype=torch.float32) 

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    input_size = X_train.shape[1]
    output_size = 1

    model = NNModel(input_size, neurons_number, output_size)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate) # safe επιλογή ο Adam, 0.001 αρχικό learning rate
    print("\n #################### " + str(neurons_number) + " neurons with " + str(optimizer.param_groups[0]['lr']) + " learning rate and " + str(epochs) + " epochs ########################## \n")
    # Αποθήκευση των hyperparameters ώστε να γίνει σύγκριση μεταξύ διαφορετικών runs στο WandB
    wandb.log({
    "neurons_number": str(neurons_number),
    "learning_rate": learning_rate,
    "epochs": epochs
    })
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        
        model.train() # εκτέλεση του training\
        optimizer.zero_grad() # μηδενισμός των gradients από το προηγούμενο βήμα
        
        y_pred = model(X_train_t) # πρόβλεψη του μοντέλου
        loss = loss_function(y_pred, y_train_t) # υπολογισμός του κόστους 
        
        loss.backward() # backpropagation για να υπολογίσουμε τα gradients των παραμέτρων του μοντέλου
        optimizer.step() # update των παραμέτρων του μοντέλου με βάση τα gradients και το learning rate
        
        # Validation
        # model.eval() # έλεγχος κατά τη διάρκεια του training, για να βλέπουμε αν το μοντέλο γενικεύει καλά
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val_t)
            val_loss = loss_function(y_val_pred, y_val_t)
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())

        if epoch % 10 == 0: # ανα 100 epochs βλέπουμε την πρόοδο στη μάθηση
            if model_name == "Fuel":
                print(f"[Fuel] Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
            else:
                print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")        

        wandb.log({
        "epoch": epoch,
        "Train_Loss": loss.item(),
        "Validation_Loss": val_loss.item()
        })
        
    # Ερώτημα 5
    # Test evaluation
    model.eval()

    # explainer = shap.DeepExplainer(model, X_train_t)
    # shap_values = explainer.shap_values(X_test_t)
    # shap.summary_plot(shap_values, X_test, 
    #               feature_names=X.columns.tolist(),
    #               plot_type="bar")
    # shap.summary_plot(shap_values, X_test,
    #               feature_names=X.columns.tolist())

    with torch.no_grad():
        y_test_pred = model(X_test_t)
        test_loss = loss_function(y_test_pred, y_test_t)

    test_rmse = math.sqrt(test_loss.item())

    if model_name == "Fuel":
        print("\nFuel Test Loss:", test_loss.item())
        print("Fuel Test RMSE:", math.sqrt(test_loss.item()))
    else:
        print("\nTest Loss:", test_loss.item())
        print("Test RMSE:", math.sqrt(test_loss.item()))

    wandb.log({
    "Test_Loss": test_loss.item(),
    "Test_RMSE": test_rmse
    })

    # Γραφήματα training και validation loss συναρτήσει των epochs
    fig,ax=plt.subplots()
    ax.plot(train_losses, label="Training Loss",color='#9671bd') #set better visual colors
    ax.plot(val_losses, label="Validation Loss",color='#77b5b6')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale('log') # log to see the waves
    ax.set_title(plot_title_loss, loc='center', y=1.1)  # move the title above the legend
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25) # transparent major grid
    ax.minorticks_on()
    ax.grid(True,which='minor', linestyle='-', linewidth=0.25, alpha=0.15) # transparent minor grid
    ax.set_axisbelow(True) # to get grid below data
    desired_order = [0, 1]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(  #more beautiful legend at the top
    [handles[i] for i in desired_order],
    [labels[i] for i in desired_order],
    loc = 'upper center',
    bbox_to_anchor = (0.5, 1.15),  # center top, above axes
    ncol = 4,                      # spread horizontally
    frameon = False                # removes legend border
    ) 
    plt.subplots_adjust(left=0.1, right=0.98, top=0.85, bottom=0.1)
    #manager = plt.get_current_fig_manager()
    #manager.window.state('zoomed')
    plot_name = f"{model_name}_loss_neurons_{neurons_number}_lr_{learning_rate}_epochs_{epochs}.png"
    plt.savefig(plot_name, dpi=300, bbox_inches="tight")
    wandb.log({"loss_plot": wandb.Image(plot_name)})
    plt.close()

    # Σύγκριση προβλέψεων με πραγματικές τιμές στο test set
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_t)

    y_test_pred_np = y_test_pred.numpy()
    y_test_np = y_test_t.numpy()
    y_test_pred_real = y_scaler.inverse_transform(y_test_pred_np).flatten()
    y_test_real = y_scaler.inverse_transform(y_test_np).flatten()
    y_test_pred_np = y_test_pred_np.flatten()
    y_test_np = y_test_np.flatten()

    fig,ax=plt.subplots()
    ax.scatter(y_test_np, y_test_pred_np, color='#9671bd', s = 60,
               edgecolors = '#6a408d',
               linewidths = 1.5,
               zorder = 3)
    ax.set_xlabel(scaled_xlabel)
    ax.set_ylabel(scaled_ylabel)
    ax.set_title(plot_title_scaled, loc='center', y=1.1)  # move the title above the legend
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25) # transparent major grid
    ax.minorticks_on()
    ax.grid(True,which='minor', linestyle='-', linewidth=0.25, alpha=0.15) # transparent minor grid
    ax.set_axisbelow(True) # to get grid below data
    plt.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.1)
    #manager = plt.get_current_fig_manager()
    #manager.window.state('zoomed')
    m, b = np.polyfit(y_test_np, y_test_pred_np, 1)
    x_line = np.linspace(min(y_test_np), max(y_test_np), 100)
    plt.plot(x_line, m * x_line + b, color='#77b5b6', label=f'Fit: y = {m:.3f}x {b:+.3f}', zorder=4)
    min_val = min(min(y_test_np), min(y_test_pred_np))
    max_val = max(max(y_test_np), max(y_test_pred_np))
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='y=x', zorder=4)
    desired_order = [0, 1]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(  #more beautiful legend at the top
    [handles[i] for i in desired_order],
    [labels[i] for i in desired_order],
    loc = 'upper center',
    bbox_to_anchor = (0.5, 1.15),  # center top, above axes
    ncol = 4,                      # spread horizontally
    frameon = False                # removes legend border
    ) 
    plot_name = f"{model_name}_predicted_vs_actual_scaled_loss_neurons_{neurons_number}_lr_{learning_rate}_epochs_{epochs}.png"
    plt.savefig(plot_name, dpi=300, bbox_inches="tight")
    wandb.log({"loss_plot": wandb.Image(plot_name)})
    plt.close()

    r2 = r2_score(y_test_np, y_test_pred_np)
    l_inf_scaled = np.max(np.abs(y_test_np - y_test_pred_np))

    # ίδιο διάγραμμα αλλά με τις πραγματικές τιμές αντί για τα scaled

    fig,ax=plt.subplots()
    ax.scatter(y_test_real, y_test_pred_real, color='#9671bd', s = 60, #set datat points parameters for better visualization
               edgecolors = '#6a408d',
               linewidths = 1.5,
               zorder = 3)
    ax.set_xlabel(real_xlabel)
    ax.set_ylabel(real_ylabel)
    ax.set_title(plot_title_real, loc='center', y=1.1)
    ax.grid(True, which='major', linestyle='-', linewidth=0.75, alpha=0.25) # transparent major grid
    ax.minorticks_on()
    ax.grid(True,which='minor', linestyle='-', linewidth=0.25, alpha=0.15) # transparent minor grid
    ax.set_axisbelow(True) # to get grid below data
    plt.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.1)
    #manager = plt.get_current_fig_manager()
    #manager.window.state('zoomed')
    m, b = np.polyfit(y_test_real, y_test_pred_real, 1)
    x_line = np.linspace(min(y_test_real), max(y_test_real), 100)
    plt.plot(x_line, m * x_line + b, color='#77b5b6', label=f'Fit: y = {m:.3f}x {b:+.3f}', zorder=4)
    min_val = min(min(y_test_real), min(y_test_pred_real))
    max_val = max(max(y_test_real), max(y_test_pred_real))
    plt.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='y=x', zorder=4)
    desired_order = [0, 1]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(  #more beautiful legend at the top
    [handles[i] for i in desired_order],
    [labels[i] for i in desired_order],
    loc = 'upper center',
    bbox_to_anchor = (0.5, 1.15),  # center top, above axes
    ncol = 4,                      # spread horizontally
    frameon = False                # removes legend border
    ) 
    plot_name = f"{model_name}_predicted_vs_actual_real_neurons_{neurons_number}_lr_{learning_rate}_epochs_{epochs}.png"
    plt.savefig(plot_name, dpi=300, bbox_inches="tight")
    wandb.log({"predicted_vs_actual_real": wandb.Image(plot_name)})
    plt.close()

    rmse_real = math.sqrt(    ((y_test_real - y_test_pred_real) ** 2).mean())

    print("Real RMSE:", rmse_real)
    print("R^2 Score:", r2)
    print(f"L_inf = {l_inf_scaled:.4f}")

    wandb.log({
    "Real_RMSE": rmse_real,
    "R2_Score": r2,
    "L_inf": l_inf_scaled
    })

    # SHAP analysis για το καλύτερο μοντέλο
    if do_shap:
        print("\nRunning SHAP analysis...")

        feature_names = X.columns.tolist()

        # Χρησιμοποιούμε μικρό sample γιατί το KernelExplainer είναι αργό
        background_size = min(200, X_train.shape[0])
        test_size = min(300, X_test.shape[0])

        np.random.seed(42)
        background_idx = np.random.choice(X_train.shape[0], background_size, replace=False)
        test_idx = np.random.choice(X_test.shape[0], test_size, replace=False)

        background = X_train[background_idx]
        X_test_shap = X_test[test_idx]

        def predict_for_shap(x):
            model.eval()
            with torch.no_grad():
                x_t = torch.tensor(x, dtype=torch.float32)
                output = model(x_t).numpy()
            return output.flatten()

        explainer = shap.KernelExplainer(predict_for_shap, background)
        shap_values = explainer.shap_values(X_test_shap)

        if isinstance(shap_values, list):
            shap_values_plot = shap_values[0]
        else:
            shap_values_plot = shap_values

        # SHAP bar plot: μέση απόλυτη επίδραση κάθε feature
        shap.summary_plot(
            shap_values_plot,
            X_test_shap,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )

        shap_bar_name = f"{model_name}_SHAP_bar_{feature_set_name}_neurons_{neurons_number}_lr_{learning_rate}_epochs_{epochs}.png"
        plt.title(f"SHAP Feature Importance - {model_name}")
        plt.tight_layout()
        plt.savefig(shap_bar_name, dpi=300, bbox_inches="tight")
        wandb.log({"SHAP_bar_plot": wandb.Image(shap_bar_name)})
        plt.close()

        # SHAP beeswarm plot: επίδραση feature value στην πρόβλεψη
        shap.summary_plot(
            shap_values_plot,
            X_test_shap,
            feature_names=feature_names,
            show=False
        )

        shap_beeswarm_name = f"{model_name}_SHAP_beeswarm_{feature_set_name}_neurons_{neurons_number}_lr_{learning_rate}_epochs_{epochs}.png"
        plt.title(f"SHAP Beeswarm Plot - {model_name}")
        plt.tight_layout()
        plt.savefig(shap_beeswarm_name, dpi=300, bbox_inches="tight")
        wandb.log({"SHAP_beeswarm_plot": wandb.Image(shap_beeswarm_name)})
        plt.close()

        print("SHAP analysis completed.")

    results_row = {
        "model_name": f"{feature_set_name}_{model_name}",
        "neurons": str(neurons_number),
        "lr": learning_rate,
        "epochs": epochs,
        "rmse_scaled": math.sqrt(test_loss.item()),
        "rmse_real": rmse_real,
        "r2": r2,
        "L_inf": l_inf_scaled
    }
    results = "results.csv"
    write_header = (not os.path.isfile(results)) or os.path.getsize(results) == 0
    with open(results, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_row.keys(), delimiter='\t')
        if write_header:
            writer.writeheader()
        writer.writerow(results_row)

# Run the simple model for Fuel Consumption
X = X_simple
feature_set_name = "Simple"
sweep_id = wandb.sweep(sweep_config_simple, project="FOC_nn")
wandb.agent(sweep_id, lambda: train_Fuel_Consumption("Simple Fuel Consumption"))

# Run the simple model for NOx
X = X_simple
feature_set_name = "Simple"
sweep_id = wandb.sweep(sweep_config_simple, project="NOx_nn")
wandb.agent(sweep_id, lambda: train_NOx("Simple NOx"))

print("\n #################### Question 6 ########################## \n")
# feature engineering
dff = df.copy()
dff["Power"] = dff["Engine Torque"] * dff["Rot. Speed"] # Physical meaning and better correlation
dff["Thermal_Load"] = dff["Engine Torque"] * dff["Exhaust Gas Temperature"] # Better correlation
dff["EGR_Torque"] = dff["Engine Torque"] * dff["EGR Command"] # Better correlation
dff["EGMF_Torque"] = dff["Engine Torque"] * dff["Exhaust Gas Mass Flow"] # Better correlation
# plt.rcParams.update({
#     'font.family': 'sans-serif',  # monospace font
#     'font.sans-serif': 'Calibri',  # specific sans-serif font
#     'font.size': 18,
#     'axes.titlesize': 18,
#     'axes.labelsize': 18,
#     'xtick.labelsize': 18,
#     'ytick.labelsize': 18,
#     'legend.fontsize': 18,
#     'figure.titlesize': 18
# }) 
# heatmap(dff.drop(columns=["rpm_group"]), "Correlation Matrix with Features")
# plt.rcParams.update({
#     'font.family': 'sans-serif',  # monospace font
#     'font.sans-serif': 'Calibri',  # specific sans-serif font
#     'font.size': 20,
#     'axes.titlesize': 20,
#     'axes.labelsize': 20,
#     'xtick.labelsize': 20,
#     'ytick.labelsize': 20,
#     'legend.fontsize': 20,
#     'figure.titlesize': 20
# }) 
X_engineered = dff.drop(columns=["NOx", "Fuel Consumption", "Intake Pressure", "lambda", "rpm_group", "Torque Reference"]) # αφαιρούμε τα μεγέθη που δεν έχουν νόημα για την πρόβλεψη των NOx (time) ή είναι πολύ άμεσα συνδεδεμένα με αυτά

# Run the simple model with engineered features for Fuel Consumption
X = X_engineered
feature_set_name = "Engineered_Simple"
sweep_id = wandb.sweep(sweep_config_simple, project="FOC_nn")
wandb.agent(sweep_id, lambda: train_Fuel_Consumption("Featured Fuel Consumption"))

# Run the simple model with engineered features for NOx
X = X_engineered
feature_set_name = "Engineered_Simple"
sweep_id = wandb.sweep(sweep_config_simple, project="NOx_nn")
wandb.agent(sweep_id, lambda: train_NOx("Featured NOx"))

# Run the complex model for Fuel Consumption
X = X_engineered
feature_set_name = "Engineered_Complex"
sweep_id = wandb.sweep(sweep_config_complex, project="FOC_nn")
wandb.agent(sweep_id, lambda: train_Fuel_Consumption("Complex Fuel Consumption"))

# Run the complex model for NOx
X = X_engineered
feature_set_name = "Engineered_Complex"
sweep_id = wandb.sweep(sweep_config_complex, project="NOx_nn")
wandb.agent(sweep_id, lambda: train_NOx("Complex NOx"))

# Final run with SHAP for the best Fuel Consumption model
X = X_engineered
feature_set_name = "Best_Engineered_Complex"

wandb.init(project="FOC_nn")
wandb.run.name = "Best_Fuel_Consumption_SHAP"

run_neural_network(
    X,
    df["Fuel Consumption"],
    "Fuel Consumption",
    500,
    [64, 64],
    0.01,
    "Actual Fuel Consumption (scaled)",
    "Predicted Fuel Consumption (scaled)",
    "Actual Fuel Consumption [kg/h]",
    "Predicted Fuel Consumption [kg/h]",
    "Training vs Validation Loss for Best Fuel Consumption Model",
    "Predicted vs Actual Fuel Consumption",
    "Predicted vs Actual Fuel Consumption",
    do_shap=True
)

wandb.finish()

# Final run with SHAP for the best NOx model
X = X_engineered
feature_set_name = "Best_Engineered_Complex"

wandb.init(project="NOx_nn")
wandb.run.name = "Best_NOx_SHAP"

run_neural_network(
    X,
    df["NOx"],
    "NOx",
    500,
    [64, 64],
    0.001,
    "Actual NOx (scaled)",
    "Predicted NOx (scaled)",
    "Actual NOx [ppm]",
    "Predicted NOx [ppm]",
    "Training vs Validation Loss for Best NOx Model",
    "Predicted vs Actual NOx",
    "Predicted vs Actual NOx",
    do_shap=True
)

wandb.finish()