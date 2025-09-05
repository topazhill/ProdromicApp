import base64
import plotly.tools
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from scipy import stats
import pandas as pd
import tabulate
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly


IMPUTING = False
FILTER = None
KS_SAMPLE_SIZE = 40  # Sample size for KS test
KL_BINS = 20  # Number of bins for KL divergence histogram
WEIGHTS = [0, 0.8, 0.6, 0.4, 0.2, 0.9, 0.9, 0]  # Weights for each column

current_tab = 'tab-1'
loaded = False

def column_data(data, index):
    if not data:
        return []
    values = [row[index] for row in data if isinstance(row[index], (int, float))]
    return values

class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message, index, pid, error_type):
        super().__init__(message)
        self.message = message
        self.index = index
        self.pid = pid
        self.error_type = error_type

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def get_from_diagnosis(data, diagnosis):
    if not data:
        return []
    return [row for row in data if row[7] == diagnosis or row[7] == 'DX']

def parse_file(filename, header=False, missing_flag=False):
    try:
        file = open(filename, 'r')
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None
    except Exception as e:
        print(f"An error occurred while opening the file: {e}")
        return None

    with open(filename, 'r') as f:
        lines = f.readlines()

    return parse_data(lines, header, missing_flag)

def parse_contents(contents, header=False, missing_flag=False):
    if contents is None:
        return None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    lines = decoded.decode('utf-8').split('\r')

    return parse_data(lines, header, missing_flag)


def parse_data(lines, header=False, missing_flag=False):
    headers = lines[0].strip().split(',')[0:8]
    lines = lines[1:]

    impute_data = []
    data = []
    missing_values = []

    if header:
        data.append(headers)

    persons = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        person = {}

        parts = line.split(',')[0:8]
        if parts == [''] * len(headers):
            continue
        for i, part in enumerate(parts):
            part = part.strip()
            if i == 0:
                person['id'] = part
            elif i == 1:
                person['roi'] = float(part) if is_float(part) else np.nan
            elif i == 2:
                person['age'] = float(part) if is_float(part) else np.nan
            elif i == 3:
                if part.isdigit():
                    person['gender'] = int(part)
                else:
                    person['gender'] = 1
                    missing_values.append(
                        ValidationError(f"Missing {headers[i]} value, patient ID: {person['id']}", i, person['id'],
                                        'Missing'))
            elif i == 4:
                person['edu'] = int(part) if part.isdigit() else np.nan
            elif i == 5:
                person['cdrsb'] = float(part) if is_float(part) else np.nan
            elif i == 6:
                person['adas'] = float(part) if is_float(part) else np.nan
            elif i == 7:
                person['dia'] = part.strip()

        for j, count in enumerate(
                [person['roi'], person['age'], person['gender'], person['edu'], person['cdrsb'], person['adas']]):
            if count is np.nan:
                missing_values.append(
                    ValidationError(f"Missing {headers[j + 1]} value, patient ID: {person['id']}", j + 1, person['id'],
                                    'Missing'))

        persons.append(person)
        impute_data.append([person['roi'], person['age'], person['edu'], person['cdrsb'], person['adas']])

    if IMPUTING:
        impute_data = np.array(impute_data)
        # Impute missing values using IterativeImputer
        imputer = IterativeImputer(max_iter=10, random_state=0)
        impute_data = imputer.fit_transform(impute_data)
        impute_data = impute_data.tolist()

    for j in range(len(impute_data)):
        persons[j]['roi'] = impute_data[j][0]
        persons[j]['age'] = impute_data[j][1]
        persons[j]['edu'] = impute_data[j][2]
        persons[j]['cdrsb'] = impute_data[j][3]
        persons[j]['adas'] = impute_data[j][4]
        data.append([persons[j]['id'],
                     persons[j]['roi'],
                     persons[j]['age'],
                     persons[j]['gender'],
                     persons[j]['edu'],
                     persons[j]['cdrsb'],
                     persons[j]['adas'],
                     persons[j]['dia']])

    if FILTER is not None:
        data = get_from_diagnosis(data, FILTER)

    return [data, missing_values] if missing_flag else [data]

'''Validation module for input data used in Alzheimer's disease prediction models.'''
class Validation:
    AGE_MIN = 50
    AGE_MAX = 100
    YEARS_EDUCATION_MIN = 0
    YEARS_EDUCATION_MAX = 20
    VALID_SEX = {1, 0}
    CDRSB_MIN = 0
    CDRSB_MAX = 6
    ADAS13_MIN = 3
    ADAS13_MAX = 40
    GREY_MATTER_MIN = -0.21
    GREY_MATTER_MAX = 0.09

    def __init__(self, plotting=False):
        print("Validation class initialized")
        self.p_id = -1

        self.training_data = parse_file("public/training data.csv", header=True)[0]
        self.headers = self.training_data[0] if self.training_data else []
        self.training_data = self.training_data[1:]  # Skip headers
        self.current_data = None
        self.plotting = plotting
        if self.plotting:
            self.pdf = matplotlib.backends.backend_pdf.PdfPages("public/validation plots.pdf")

    def plot(self, data_point, index):

        values = [row[index] for row in self.training_data if isinstance(row[index], (int, float))]
        plt.figure(figsize=(10, 5))

        height = 0
        if self.current_data is not None:
            test_values = [row[index] for row in self.current_data if isinstance(row[index], (int, float))]
            sns.histplot(test_values, color='red', label='Test Data', kde=True, stat='density', bins=30)
            heights = plt.gca().get_lines()[0].get_data()

            for j in range(len(heights[0]) - 1):
                if heights[0][j] <= data_point < heights[0][j + 1]:
                    height = heights[1][j]
                    break
            if data_point == heights[0][-1]:
                height = heights[1][-1]

        plt.scatter(data_point, height, color='red', s=100, label='Data Point', zorder=5, marker='x')
        sns.histplot(values, color='blue', label='Training Data', kde=True, stat='density', bins=30)
        plt.title(f'Distribution of {self.headers[index] if self.headers else f"Column {index}"}: Patient ID {self.p_id}')
        plt.xlabel(self.headers[index] if self.headers else f'Column {index}')
        plt.ylabel('Density')
        plt.legend()
        self.pdf.savefig()
        plt.close()

        plt.figure(figsize=(10, 5))

        height = 0
        if self.current_data is not None:
            test_values = [row[index] for row in self.current_data if isinstance(row[index], (int, float))]
            sns.ecdfplot(test_values, color='red', label='Test Data')
            heights = plt.gca().get_lines()[0].get_data()

            for j in range(len(heights[0]) - 1):
                if heights[0][j] <= data_point < heights[0][j + 1]:
                    height = heights[1][j]
                    break
            if data_point == heights[0][-1]:
                height = heights[1][-1]

        plt.scatter(data_point, height, color='red', s=100, label='Data Point', zorder=5, marker='x')
        sns.ecdfplot(values, color='blue', label='Training Data')
        plt.title(f'Distribution of {self.headers[index] if self.headers else f"Column {index}"}: Patient ID {self.p_id}')
        plt.xlabel(self.headers[index] if self.headers else f'Column {index}')
        plt.ylabel('Density')
        plt.legend()
        self.pdf.savefig()
        plt.close()

    def validate(self, data):
        if FILTER is not None:
            data = get_from_diagnosis(data, FILTER)
        error_flags = []
        nan_flags = []
        if len(data) >= 20:
            self.current_data = data
        else:
            self.current_data = None

        """Check for data drift."""
        results, sig_results = self.data_drift_check()
        """Calculate KL divergence between training and test data distributions."""
        if self.current_data is not None:
            div_results = self.KL_divergence(self.training_data, self.current_data, self.headers)

        """Run all validations and return error flags."""
        print("Running validations on individual data...")
        for p in data:
            self.p_id = p[0]
            self.validate_age(p[2], error_flags, nan_flags)
            self.validate_YoE(p[4], error_flags, nan_flags)
            self.validate_sex(p[3], nan_flags)
            self.validate_cdr(p[5], error_flags, nan_flags)
            self.validate_adas13(p[6], error_flags, nan_flags)
            self.validate_grey_matter(p[1], error_flags, nan_flags)

        if self.plotting:
            self.pdf.close()
            print("Validation plots saved to validation plots.pdf")

        return error_flags, nan_flags, results, sig_results, div_results if self.current_data is not None else []

    def data_drift_check(self):
        """Check for data drift using Kolmogorov-Smirnov test on a random sample."""
        print("Checking for data drift using Kolmogorov-Smirnov test...")
        results = []
        sig_results = []
        significant = []
        for i in range(1, len(self.headers) - 1):
            if self.current_data is not None:
                test_values = column_data(self.current_data, i)
                training_values = column_data(self.training_data, i)

                test_frame = pd.DataFrame(test_values, columns=[self.headers[i]])
                test_sample = test_frame.sample(n=min(KS_SAMPLE_SIZE, len(test_frame)), random_state=42) if len(test_frame) > 100 else test_frame
                training_frame = pd.DataFrame(training_values, columns=[self.headers[i]])
                training_sample = training_frame.sample(n=min(KS_SAMPLE_SIZE, len(training_frame)), random_state=42) if len(training_frame) > 100 else training_frame

                if test_sample.empty or training_sample.empty:
                    print(f"No data available for {self.headers[i]}")
                    continue

                ks_statistic, p_value = stats.ks_2samp(test_sample[self.headers[i]], training_sample[self.headers[i]], nan_policy='omit')
                if p_value < 0.05:
                    significant.append((self.headers[i], ks_statistic, p_value))
                print(f"KS Statistic for {self.headers[i]}: {ks_statistic}, p-value: {p_value}")
                results.append([self.headers[i], ks_statistic, p_value])

            else:
                print(f"No data available for {self.headers[i]}")
        print(" ")
        if significant:
            print("Significant drift detected in the following columns:")
            for col, stat, p in significant:
                print(f"{col}: KS Statistic = {stat}, p-value = {p}")
                sig_results.append([col, stat, p])
        else:
            print("No significant drift detected.")
        print(" ")
        return results, sig_results

    def KL_divergence(self, training_data, test_data, headers):
        """Calculate KL divergence between training and test data distributions."""
        print("Calculating KL divergence between training and test data distributions...")
        kl_divergences = {}
        div_results = []
        for i in range(1, len(headers) - 1):
            train_values = column_data(training_data, i)
            test_values = column_data(test_data, i)

            if not train_values or not test_values:
                print(f"No data available for {headers[i]}")
                continue

            # Create histograms
            min_value = min(min(train_values), min(test_values))
            max_value = max(max(train_values), max(test_values))
            bins = np.linspace(min_value, max_value, KL_BINS)

            train_hist, _ = np.histogram(train_values, bins=bins, density=True)
            test_hist, _ = np.histogram(test_values, bins=bins, density=True)

            # Add a small constant to avoid division by zero
            train_hist += 1e-10
            test_hist += 1e-10

            kl_div = stats.entropy(train_hist, test_hist)
            kl_divergences[headers[i]] = kl_div
            print(f"KL Divergence for {headers[i]}: {kl_div}")
            div_results.append([headers[i], kl_div])
        print(" ")
        return div_results

    def validate_age(self, age, error_flags, nan_flags):
        if age is np.nan:
            return
        if not isinstance(age, (int, float)):
            nan_flags.append(ValidationError(f'Age {age} is not a number, patient ID: {self.p_id}', 2, self.p_id, 'NaN'))
            return
        """Validate age against predefined limits."""
        if not (self.AGE_MIN <= age <= self.AGE_MAX):
            error_flags.append(ValidationError(f'Age {age} out of range: {"Above" if age >= self.AGE_MAX else "Below"}, patient ID: {self.p_id}', 2, self.p_id, "AD" if age >= self.AGE_MAX else "CN"))
            if self.plotting:
                self.plot(age, 2)
        elif not (0 <= age <= 120):
            nan_flags.append(ValidationError(f'Age {age} is not a valid age, patient ID: {self.p_id}', 2, self.p_id, 'NaN'))

    def validate_YoE(self, YoE, error_flags, nan_flags):
        if YoE is np.nan:
            return
         # Validate years of education against predefined limits.
        if not isinstance(YoE, (int, float)):
            nan_flags.append(ValidationError(f'Years of Education {YoE} is not a number, patient ID: {self.p_id}', 3, self.p_id, 'NaN'))
            return
        if not (self.YEARS_EDUCATION_MIN <= YoE <= self.YEARS_EDUCATION_MAX):
            error_flags.append(ValidationError(f'Years of Education {YoE} out of range: {"Above" if YoE >= self.YEARS_EDUCATION_MAX else "Below"}, patient ID: {self.p_id}', 3, self.p_id, "Above" if YoE >= self.YEARS_EDUCATION_MAX else "Below"))
            if self.plotting:
                self.plot(YoE, 4)
        elif not (0 <= YoE <= 30):
            nan_flags.append(ValidationError(f'Years of Education {YoE} is not a valid number, patient ID: {self.p_id}', 3, self.p_id, 'NaN'))

    def validate_sex(self, sex, nan_flags):
        if sex not in self.VALID_SEX:
            nan_flags.append(ValidationError(f'Sex {sex} is not valid, patient ID: {self.p_id}', 4, self.p_id, 'NaN'))

    def validate_cdr(self, cdr, error_flags, nan_flags):
        if cdr is np.nan:
            return
         # Validate CDR Sum of Boxes against predefined limits.
        if not isinstance(cdr, float):
            nan_flags.append(ValidationError(f'CDR Sum of Boxes {cdr} is not a valid integer, patient ID: {self.p_id}', 5, self.p_id, 'NaN'))
            return
        if not (self.CDRSB_MIN <= cdr <= self.CDRSB_MAX):
            error_flags.append(ValidationError(f'CDR Sum of Boxes {cdr} out of range: {"Above" if cdr >= self.CDRSB_MAX else "Below"}, patient ID: {self.p_id}', 5, self.p_id, "AD" if cdr >= self.CDRSB_MAX else "CN"))
            if self.plotting:
                self.plot(cdr, 5)
        elif not (0 <= cdr <= 18):
            nan_flags.append(ValidationError(f'CDR Sum of Boxes {cdr} is not a valid number, patient ID: {self.p_id}', 5, self.p_id, 'NaN'))

    def validate_adas13(self, adas13, error_flags, nan_flags):
        if adas13 is np.nan:
            return
         # Validate ADAS13 against predefined limits.
        if not isinstance(adas13, float):
            nan_flags.append(ValidationError(f'ADAS13 {adas13} is not a valid integer, patient ID: {self.p_id}', 6, self.p_id, 'NaN'))
            return
        if not (self.ADAS13_MIN <= adas13 <= self.ADAS13_MAX):
            error_flags.append(ValidationError(f'ADAS13 {adas13} out of range: {"Above" if adas13 >= self.ADAS13_MAX else "Below"}, patient ID: {self.p_id}', 6, self.p_id, "AD" if adas13 >= self.ADAS13_MAX else "CN"))
            if self.plotting:
                self.plot(adas13, 6)
        elif not (0 <= adas13 <= 85):
            nan_flags.append(ValidationError(f'ADAS13 {adas13} is not a valid number, patient ID: {self.p_id}', 6, self.p_id, 'NaN'))

    def validate_grey_matter(self, grey_matter, error_flags, nan_flags):
        if grey_matter is np.nan:
            return
         # Validate grey matter against predefined limits.
        if not isinstance(grey_matter, float):
            nan_flags.append(ValidationError(f'Grey matter {grey_matter} is not a valid integer, patient ID: {self.p_id}', 1, self.p_id, 'NaN'))
            return
        if not (self.GREY_MATTER_MIN <= grey_matter <= self.GREY_MATTER_MAX):
            error_flags.append(ValidationError(f'Grey matter {grey_matter} out of range: {"Above" if grey_matter >= self.GREY_MATTER_MAX else "Below"}, patient ID: {self.p_id}', 1, self.p_id, 'CN' if grey_matter >= self.GREY_MATTER_MAX else "AD"))
            if self.plotting:
                self.plot(grey_matter, 1)
        elif not (-0.5 <= grey_matter <= 0.5):
            nan_flags.append(ValidationError(f'Grey matter {grey_matter} is not a valid number, patient ID: {self.p_id}', 1, self.p_id, 'NaN'))


def evaluate(test_data, filename, errors, nan, missing):
    """Evaluate the validation results and print summary."""
    if errors or nan or missing:
        total = errors + nan + missing
        weights = WEIGHTS

        print(f"Total errors found: {len(total)}")
        print(f"Out of range values: {len(errors)}")
        print(f"NaN values: {len(nan)}")
        print(f"Missing {"imputed" if IMPUTING else ""} values: {len(missing)}")

        patients = {}
        columns = [[] for _ in range(len(test_data[0]))]
        for error in total:
            if error.pid not in patients:
                patients[error.pid] = [0, []]
            patients[error.pid][0] += weights[error.index]
            patients[error.pid][1].append(error)
            columns[error.index].append(error)

        with open(f'errors bin/{filename} errors.txt', 'w') as f:
            for pid, [_, errs] in sorted(patients.items(), key=lambda x: x[1][0], reverse=True):
                above_count = 0
                f.write(f"Patient ID: {pid}\n")
                for error in errs:
                    f.write(f"  {error.message} (Column {error.index})\n")
                    if error.error_type == "AD":
                        above_count += 1
                    elif error.error_type == "CN":
                        above_count -= 1

                if above_count > len(errs) - 1:
                    f.write("  Overall trend: Above normal MCI range, towards AD\n")
                elif above_count < -len(errs) + 1:
                    f.write("  Overall trend: Below normal MCI range, towards CN\n")
                else:
                    f.write("  Overall trend: Mixed deviations\n")
                f.write("\n")

        print(f"Validation errors written to error bin/'{filename} errors.txt'")

        print("Errors by column:")
        table = []
        for i in range(1, len(test_data[0]) - 1):
            table.append([headers[i], len(columns[i]), f"{100 - (len(columns[i]) / len(test_data) * 100):.2f}%"])
        print(tabulate.tabulate(table, headers=["Column", "Number of Errors", "Percentage of Valid Data"], tablefmt="grid"))
        print(" ")

        errored_records = set(error.pid for error in total)
        print(f"Percentage of error-free records: {100 - len(errored_records) / len(test_data) * 100:.2f}%")

        num_points = len(test_data) * (len(test_data[0]) - 2)  # Exclude ID and Diagnosis columns

        no_dupes = set((error.pid, error.index) for error in total)
        percentage = 100 - (len(no_dupes)) / num_points * 100
        print(f"Percentage of valid data points: {percentage:.2f}%")
        print(" ")

        patient_weights = {}
        for error in total:
            patient_weights[error.pid] = min(patient_weights.get(error.pid, 0) + weights[error.index], 1)

        weighted_percentage = 100 - (sum(patient_weights.values()) / len(test_data)) * 100
        print(f"Data quality: {weighted_percentage:.2f}%")

        if weighted_percentage >= 90:
            print("Gold: Data quality is acceptable. ")
        elif weighted_percentage >= 80:
            print("Silver: Data quality is moderate, consider reviewing the data before use. ")
        else:
            print("Bronze: Data quality is poor, significant quantity of issues found. Review the data before use.")

    else:
        print("All validations passed. Data quality is excellent.")

# Initialize validation
validation = Validation()
test_data = []
headers = []

# Initialize Dash app
app = Dash(suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.H1("Data Validation Dashboard"),
    dcc.Checklist(
        id='options-checklist',
        options=[{'label': 'Enable Missing Data Imputation', 'value': 'impute'}, {'label': 'Filter by MCI Diagnosis', 'value': 'filter'}],
        value=['impute' if IMPUTING else "", 'filter' if FILTER == 'MCI' else ""],
        style={'margin': '10px'},
        inline=True
    ),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-data-upload'),
], style={'horizontalAlign': 'center', 'width': '80%', 'margin': 'auto'})

@app.callback(Output('output-data-upload', 'children'),
            Input('upload-data', 'contents'),
            Input('options-checklist', 'value'),
            State('upload-data', 'filename'))
def update_output(content, checklist, filename):
    global IMPUTING, FILTER, test_data, headers, loaded
    IMPUTING = 'impute' in checklist
    FILTER = 'MCI' if 'filter' in checklist else None
    loaded = False
    if content is not None and filename.endswith('.csv'):
        test_data, missing = parse_contents(content, header=True, missing_flag=True)
        headers = test_data[0] if test_data else []
        test_data = test_data[1:]  # Skip headers
        if FILTER:
            test_data = get_from_diagnosis(test_data, FILTER)
        errors, nan, results, sig_results, kl_results = validation.validate(test_data)

        total = errors + nan + missing
        weights = WEIGHTS

        print(f"Total errors found: {len(total)}")
        print(f"Out of range values: {len(errors)}")
        print(f"NaN values: {len(nan)}")
        print(f"Missing {"imputed" if IMPUTING else ""} values: {len(missing)}")

        patients = {}
        columns = [[] for _ in range(len(test_data[0]))]
        for error in total:
            if error.pid not in patients:
                patients[error.pid] = [0, []]
            patients[error.pid][0] += weights[error.index]
            patients[error.pid][1].append(error)
            columns[error.index].append(error)

        table = []
        for i in range(1, len(test_data[0]) - 1):
            table.append([headers[i], len(columns[i]), f"{100 - (len(columns[i]) / len(test_data) * 100):.2f}%"])

        errored_records = set(error.pid for error in total)
        error_free = 100 - len(errored_records) / len(test_data) * 100

        num_points = len(test_data) * (len(test_data[0]) - 2)  # Exclude ID and Diagnosis columns

        no_dupes = set((error.pid, error.index) for error in total)
        percentage = 100 - (len(no_dupes)) / num_points * 100

        patient_weights = {}
        for error in total:
            patient_weights[error.pid] = min(patient_weights.get(error.pid, 0) + weights[error.index], 1)

        weighted_percentage = 100 - (sum(patient_weights.values()) / len(test_data)) * 100
        quality_level = "Gold" if weighted_percentage >= 90 else "Silver" if weighted_percentage >= 80 else "Bronze"
        message = "Data quality is acceptable." if quality_level == "Gold" else "Data quality is moderate, consider reviewing the data before use." if quality_level == "Silver" else "Data quality is poor, significant quantity of issues found. Review the data before use."

        return html.Div([
            html.H2("Current File: " + filename),
            dcc.Tabs(id='tabs', value=current_tab, children=[
                dcc.Tab(label='Error Summary', value='tab-1', children=[
                    html.Br(),
                    html.Div([
                        html.H1("Error Summary"),
                        html.H2(f"Data quality: {weighted_percentage:.2f}%"),
                        html.H2(f"Quality Level: {quality_level}"),
                        html.P(message),
                        html.Br(),
                        html.P(f"Total errors found: {len(total)}"),
                        html.P(f"Out of range values: {len(errors)}"),
                        html.P(f"NaN values: {len(nan)}"),
                        html.P(f"Missing {'imputed' if IMPUTING else ''} values: {len(missing)}"),
                        html.P(f"Percentage of error-free records: {error_free:.2f}%"),
                        html.P(f"Percentage of valid data points: {percentage:.2f}%"),
                        html.Br(),
                    ], style={'width': '60%', 'margin': 'auto', 'textAlign': 'center', 'background': '#FFD700' if quality_level == "Gold" else '#C0C0C0' if quality_level == "Silver" else '#CD7F32', 'padding': '20px', 'borderRadius': '10px'}),
                    html.H3("Errors by Column"),
                    dash_table.DataTable(
                        id='summary-table',
                        columns=[{"name": i, "id": i} for i in ["Column", "Number of Errors", "Percentage of Valid Data"]],
                        data=[{"Column": row[0], "Number of Errors": row[1], "Percentage of Valid Data": row[2]} for row in table],
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '5px 10px'},
                        style_cell_conditional=[
                            {'if': {'column_id': 'Number of Errors'}, 'textAlign': 'right'},
                            {'if': {'column_id': 'Percentage of Valid Data'}, 'textAlign': 'right'}
                        ]
                    ),
                    html.Br(),

                    html.H3("All Errors"),
                    dash_table.DataTable(
                        id='error-table',
                        columns=[{"name": i, "id": i} for i in ["Patient ID", "Column", "Error Type", "Message"]],
                        data=[{"Patient ID": int(err.pid), "Column": headers[err.index] if headers else f"Column {err.index}", "Error Type": err.error_type, "Message": err.message.split(",")[0]} for err in errors + nan + missing],
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '5px 10px'},
                        style_cell_conditional=[
                            {'if': {'column_id': 'Patient ID'}, 'textAlign': 'right', 'margin': '10px 10px'}
                        ]
                    )
                ]),
                dcc.Tab(label='Drift Analysis', value='tab-3', children=[
                    html.Div([
                        html.H1("Drift Analysis"),
                        html.P("Results of Kolmogorov-Smirnov tests and KL divergence calculations to assess data drift."),
                        html.Br(),
                        html.H3("Kolmogorov-Smirnov Test Results"),
                        dash_table.DataTable(
                            id='ks-table',
                            columns=[{"name": i, "id": i} for i in ["Column", "KS Statistic", "p-value"]],
                            data=[{"Column": row[0], "KS Statistic": f"{row[1]:.4f}", "p-value": f"{row[2]:.4f}"} for row in results],
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'padding': '5px 10px'},
                            style_cell_conditional=[
                                {'if': {'column_id': 'KS Statistic'}, 'textAlign': 'right'},
                                {'if': {'column_id': 'p-value'}, 'textAlign': 'right'}
                            ]
                        ),
                        html.Br(),
                        html.H3("Significant KS Results"),
                        dash_table.DataTable(
                            id='sig-drift-table',
                            columns=[{"name": i, "id": i} for i in ["Column", "KS Statistic", "p-value"]],
                            data=[{"Column": row[0], "KS Statistic": f"{row[1]:.4f}", "p-value": f"{row[2]:.4f}"} for row in sig_results],
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'padding': '5px 10px'},
                            style_cell_conditional=[
                                {'if': {'column_id': 'KS Statistic'}, 'textAlign': 'right'},
                                {'if': {'column_id': 'p-value'}, 'textAlign': 'right'}
                            ]
                        ),

                        html.Br(),
                        html.H3("KL Divergence Results"),
                        dash_table.DataTable(
                            id='kl-table',
                            columns=[{"name": i, "id": i} for i in ["Column", "KL Divergence"] ],
                            data=[{"Column": row[0], "KL Divergence": f"{row[1]:.4f}"} for row in kl_results],
                            page_size=10,
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'left', 'padding': '5px 10px'},
                            style_cell_conditional=[
                                {'if': {'column_id': 'KL Divergence'}, 'textAlign': 'right'}
                            ]
                        ),
                    ], style={'width': '60%', 'margin': 'auto', 'textAlign': 'center'}),
                ]),
                dcc.Tab(label='Data Overview', value='tab-2', children=[
                    html.Div([
                        html.H1("Data Overview"),
                        html.P("View the dataset and explore distributions of key variables."),
                        html.Br(),
                    ], style={'width': '60%', 'margin': 'auto', 'textAlign': 'center'}),
                    dash_table.DataTable(
                        id='data-table',
                        columns=[{"name": i, "id": i} for i in headers],
                        data=[{headers[j]: row[j] for j in range(len(headers))} for row in test_data],
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'left', 'padding': '5px 10px'},
                    ),
                    html.Br(),
                    html.Div([dcc.Dropdown(options=[
                        {'label': 'ROI', 'value': 0},
                        {'label': 'Age', 'value': 1},
                        {'label': 'Gender', 'value': 2},
                        {'label': 'Years of Education', 'value': 3},
                        {'label': 'CDR Sum of Boxes', 'value': 4},
                        {'label': 'ADAS13', 'value': 5},
                    ], value=0, id='view-selector'),
                    dcc.Dropdown(options=[
                        {'label': 'Histogram', 'value': 'histogram'},
                        {'label': 'CDF', 'value': 'cdf'},
                        {'label': 'Percentile Comparison', 'value': 'percentile'}
                    ], value='histogram', id='plot-type'),
                    dcc.Dropdown(options=[patient[0] for patient in test_data], id='patient-selector'),
                    html.Br(),
                    dcc.Graph(id='data-graph', figure = {}),
                    html.Br(),
                    ], style={'width': '60%', 'margin': 'auto'}),
                ]),
            ]),
        ])
    else:
        return html.Div([
            html.H3("Upload a CSV file to see validation results.")
        ], style={'textAlign': 'center'})

@app.callback(
    Output('data-graph', 'figure'),
    Input('view-selector', 'value', allow_optional=True),
    [Input('plot-type', 'value', allow_optional=True)],
    Input('patient-selector', 'value', allow_optional=True),
)
def update_graph(selected_view, plot_type, selected_patient):
    global test_data, validation, headers
     # Set default values if inputs are None
    if not selected_view:
        selected_view = 0
    if not plot_type:
        plot_type = 'histogram'
    if not selected_patient:
        selected_patient = None
     # Generate the appropriate plot based on user selections
    if not test_data:
        return {}
    i = selected_view + 1 # Adjust for ID column
    train_values = column_data(validation.training_data, i)
    test_values = column_data(test_data, i)

    if plot_type == 'histogram':
        bins = 60
        if FILTER:
            bins = 30
        fig = plotly.graph_objects.Figure()
        fig.add_trace(plotly.graph_objects.Histogram(x=train_values, name='Training Data', opacity=0.75, nbinsx=30, histnorm='probability density'))
        fig.add_trace(plotly.graph_objects.Histogram(x=test_values, name='Test Data', opacity=0.75, nbinsx=bins, histnorm='probability density'))
        if selected_patient:
            patient_row = next((row for row in test_data if row[0] == selected_patient), None)
            if patient_row and isinstance(patient_row[i], (int, float)):
                fig.add_trace(plotly.graph_objects.Scatter(x=[patient_row[i]], y=[0], mode='markers', marker=dict(color='red', size=12, symbol='x'), name='Selected Patient'))
        fig.update_layout(barmode='overlay', title=f'Histogram of {headers[i]}', xaxis_title=headers[i], yaxis_title='Count')

    elif plot_type == 'cdf':
        fig = plotly.graph_objects.Figure()
        fig.add_trace(plotly.graph_objects.Scatter(x=np.sort(train_values), y=np.arange(1, len(train_values)+1) / len(train_values), mode='lines', name='Training Data'))
        fig.add_trace(plotly.graph_objects.Scatter(x=np.sort(test_values), y=np.arange(1, len(test_values)+1) / len(test_values), mode='lines', name='Test Data'))
        if selected_patient:
            patient_row = next((row for row in test_data if row[0] == selected_patient), None)
            if patient_row and isinstance(patient_row[i], (int, float)):
                rank = (np.sum(np.array(train_values) <= patient_row[i]) + np.sum(np.array(test_values) <= patient_row[i])) / (len(train_values) + len(test_values))
                fig.add_trace(plotly.graph_objects.Scatter(x=[patient_row[i]], y=[rank], mode='markers', marker=dict(color='red', size=12, symbol='x'), name='Selected Patient'))
        fig.update_layout(title=f'Cumulative Distribution of {headers[i]}', xaxis_title=headers[i], yaxis_title='Cumulative Probability')

    elif plot_type == 'percentile':
        percentiles_train = np.percentile(train_values, np.arange(0, 101, 1))
        percentiles_test = np.percentile(test_values, np.arange(0, 101, 1))
        fig = plotly.graph_objects.Figure()
        fig.add_trace(plotly.graph_objects.Scatter(x=np.arange(0, 101, 1), y=percentiles_train, mode='lines+markers', name='Training Data'))
        fig.add_trace(plotly.graph_objects.Scatter(x=np.arange(0, 101, 1), y=percentiles_test, mode='lines+markers', name='Test Data'))
        if selected_patient:
            patient_row = next((row for row in test_data if row[0] == selected_patient), None)
            if patient_row and isinstance(patient_row[i], (int, float)):
                rank = (np.sum(np.array(train_values) <= patient_row[i]) + np.sum(np.array(test_values) <= patient_row[i])) / (len(train_values) + len(test_values)) * 100
                fig.add_trace(plotly.graph_objects.Scatter(x=[rank], y=[patient_row[i]], mode='markers', marker=dict(color='red', size=12, symbol='x'), name='Selected Patient'))
        fig.update_layout(title=f'Percentile Comparison of {headers[i]}', xaxis_title='Percentile', yaxis_title=headers[i])
    else:
        fig = {}
    return fig

@app.callback(
    Input('tabs', 'value')
)
def switch_tab(tab):
    global current_tab
    current_tab = tab

if __name__ == '__main__':
    app.run(debug=True)

