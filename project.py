import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

np.random.seed(16920118)
random_num = 16920118

rmp_num = pd.read_csv('rmpCapstoneNum.csv', names = [1,2,3,4,5,6,7,8])
rmp_tags = pd.read_csv('rmpCapstoneTags.csv', names = np.arange(1, 21))
rmp_Qual = pd.read_csv('rmpCapstoneQual.csv',names = [1,2,3])

def Skewness_check(data):
    skewness = stats.skew(data)
    if -0.5 <= skewness <= 0.5:
        print("Acceptable skewness")
    elif -1.0 <= skewness < -0.5 or 0.5 < skewness <= 1.0:
        print("Moderately Skewed")
    else:
        print("Highly Skewed")

def cohens_d(group1, group2):
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    std1 = np.std(group1, ddof=1)
    std2 = np.std(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((std1 ** 2) + (std2 ** 2)) / 2)

    # Cohen's d
    d = (mean1 - mean2) / pooled_std
    return d
def cliffs_delta(x, y):
    """
    Calculate Cliff's Delta, a non-parametric measure of effect size.
    
    Parameters:
    - x, y: lists or arrays representing two independent groups of data.
    
    Returns:
    - delta: Cliff's Delta value.
    """
    m, n = len(x), len(y)
    u, _ = stats.mannwhitneyu(x, y, alternative='two-sided')
    delta = (2 * u) / (m * n) - 1
    effect = ""
    if(abs(delta)<0.147):
        effect = "Negligible effect"
    elif(abs(delta)<0.33):
        effect = "small effect"
    elif(abs(delta)<0.474):
        effect = "median effect"
    else:
        effect = "large effect"
    return delta, effect
def weighted_data(data, weight):
    """
    Transform data to weighted data by repeating values according to the weights.
    
    Parameters:
    data (list or array): The data values.
    weight (list or array): The corresponding weights (number of repetitions for each value).
    
    Returns:
    list: The transformed weighted data.
    """
    weighted_data = []
    for d, w in zip(data, weight):
        weighted_data.extend([d] * int(w))  # Repeat each value `w` times
    return weighted_data
def mad(data):
    median_data = np.median(data)
    deviations = np.abs(data - median_data)
    return np.median(deviations)

def row_wise(dataframe):
    rows_to_keep = ~dataframe.isnull().any(axis=1)  
    cleaned_dataframe = dataframe[rows_to_keep]
    return cleaned_dataframe
def f_test(data1, data2):
    from scipy.stats import f
    # Calculate the variance of the two datasets
    var1 = np.var(data1, ddof=1)  # Sample variance (ddof=1)
    var2 = np.var(data2, ddof=1)
    
    # Determine the F statistic
    F = var1 / var2 if var1 > var2 else var2 / var1
    
    # Degrees of freedom
    dfn = len(data1) - 1  # Degrees of freedom for numerator
    dfd = len(data2) - 1  # Degrees of freedom for denominator
    
    # Compute the p-value
    p_value = 2 * min(f.cdf(F, dfn, dfd), 1 - f.cdf(F, dfn, dfd))
    
    return F, p_value

# Question 1

# pick up the rating of professors whose gender can be determined
rating_gender = rmp_num[[1,3,7,8]][(rmp_num[7] == 1) ^ (rmp_num[8] == 1)]
# group the data based on gender
rating_male = rating_gender[rating_gender[7]==1]
rating_female = rating_gender[rating_gender[8]==1]
# transform average rating data to weighted data according to the # of rating
rating_male_weighted = weighted_data(rating_male[1], rating_male[3])
rating_female_weighted = weighted_data(rating_female[1], rating_female[3])
#Skewness_check
Skewness_check(rating_male_weighted)
Skewness_check(rating_female_weighted)
plt.figure(figsize=(8, 6))
sns.kdeplot(rating_male_weighted, shade=True,label='Male Ratings', color='blue')
sns.kdeplot(rating_female_weighted, shade=True,label='Female Ratings', color='orange')
plt.title('KDE of weighted rating Plot')
plt.xlabel('Rating Value')
plt.ylabel('Density')
plt.legend()
plt.show()
# Apply U test since the data is skewed
stats_Q1,p_value_Q1 = stats.mannwhitneyu(rating_male_weighted,rating_female_weighted,alternative='greater')
print(f"U-statistic: {stats_Q1}")
print(f"P-value: {p_value_Q1}")
# effect_size
effect_size_Q1 = cliffs_delta(rating_male_weighted,rating_female_weighted)
print(f"Cliff's Delta: {effect_size_Q1}")
print()

# Question 2
stat_levene, p_levene = stats.levene(rating_male_weighted, rating_female_weighted,center = 'median')
print(f"Levene's Test statistic: {stat_levene}")
print(f"P-value for Levene's test: {p_levene}")
# boxplot
plt.figure(figsize=(8, 6))
plt.boxplot([rating_male_weighted, rating_female_weighted], vert=True, patch_artist=True, 
            boxprops=dict(facecolor='orange', color='blue'),
            whiskerprops=dict(color='blue'),
            flierprops=dict(markerfacecolor='red', marker='o', markersize=7),
            medianprops=dict(color='blue', linewidth=2))
plt.title('Box Plot of Ratings by Gender')
plt.xlabel('Gender')
plt.ylabel('Rating Value')
plt.xticks([1, 2], ['Male', 'Female'])
plt.show()
male_rating = mad(rating_male_weighted)
female_rating = mad(rating_female_weighted)
print(f'MAD for male_rating: {male_rating}')
print(f'MAD for female_rating: {female_rating}')
print()

# Question 3
# Apply bootstrap to compute the CI for the difference in median and MAD
bootstrap_diffs_median_Q3 = []
bootstrap_diffs_MAD_Q3 = []
n_iterations = 10000
for _ in range(n_iterations):
    # Resample each group independently
    sample1 = np.random.choice(rating_male_weighted, size=len(rating_male_weighted), replace=True)
    sample2 = np.random.choice(rating_female_weighted, size=len(rating_female_weighted), replace=True)
    # Compute the difference for the resampled data
    bootstrap_diffs_median_Q3.append(np.mean(sample1)-np.mean(sample2))
    bootstrap_diffs_MAD_Q3.append(np.std(sample1,ddof= 1)-np.std(sample2,ddof = 1))
# Calculate 95% CI
lower1_Q3 = np.percentile(bootstrap_diffs_median_Q3, 2.5)
upper1_Q3 = np.percentile(bootstrap_diffs_median_Q3, 97.5)
print(f"95% CI for the difference in medians: [{lower1_Q3}, {upper1_Q3}]")
lower2_Q3 = np.percentile(bootstrap_diffs_MAD_Q3, 2.5)
upper2_Q3 = np.percentile(bootstrap_diffs_MAD_Q3, 97.5)
print(f"95% CI for the difference in MAD: [{lower2_Q3}, {upper2_Q3}]")
# Plot the bootstrap distribution for median difference
plt.figure(figsize=(10, 6))
sns.kdeplot(bootstrap_diffs_median_Q3, color='skyblue', alpha=0.7, label='Bootstrap Distribution')
plt.axvline(lower1_Q3, color='red', linestyle='--', label=f'2.5th Percentile (Lower CI: {lower1_Q3:.3f})')
plt.axvline(upper1_Q3, color='red', linestyle='--', label=f'97.5th Percentile (Upper CI: {upper1_Q3:.3f})')
plt.title('Bootstrap Distribution with 95% Confidence Interval', fontsize=16)
plt.xlabel('Difference in Medians', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()
# Plot the bootstrap distribution for MAD difference
plt.figure(figsize=(10, 6))
sns.kdeplot(bootstrap_diffs_MAD_Q3, color='skyblue', alpha=0.7, label='Bootstrap Distribution')
plt.axvline(lower2_Q3, color='red', linestyle='--', label=f'2.5th Percentile (Lower CI: {lower2_Q3:.3f})')
plt.axvline(upper2_Q3, color='red', linestyle='--', label=f'97.5th Percentile (Upper CI: {upper2_Q3:.3f})')
plt.title('Bootstrap Distribution with 95% Confidence Interval', fontsize=16)
plt.xlabel('Difference in MAD', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()
print()

# Question 4
# group the rmp_tags data based on gender and previous data preprocessing
rmp_tags_male = rmp_tags.loc[rating_male.index]
rmp_tags_female = rmp_tags.loc[rating_female.index]
# normalize the tags to tags_frequency
rmp_tags_female_normalized = rmp_tags_female.div(rating_female[3], axis=0)
rmp_tags_male_normalized = rmp_tags_male.div(rating_male[3], axis=0)
# Skewnees Check for all groups of data
ifskewnees_female = all([abs(stats.skew(rmp_tags_female_normalized[i].values))>1 for i in rmp_tags_female_normalized.columns])
ifskewnees_male = all([abs(stats.skew(rmp_tags_male_normalized[i].values))>1 for i in rmp_tags_male_normalized.columns])
print("female groups are all skewed",ifskewnees_female)
print("male groups are all skewed",ifskewnees_male)
# iterate the test over 20 tags comparisons
statistics_Q4 = []
p_value_Q4 = []
effect_size_Q4 = []
for i in range(1, 21):
    # Mann-Whitney U test if skewness is significant
    stat, p = stats.mannwhitneyu(rmp_tags_male_normalized[i], rmp_tags_female_normalized[i], alternative='two-sided')
    statistics_Q4.append(stat)
    effect_size_Q4.append(cliffs_delta(rmp_tags_male_normalized[i], rmp_tags_female_normalized[i])[0])
    p_value_Q4.append(p)
# Table plot
plt.figure(figsize=(10, 6))
index_names_tag = [
    "Tough grader", "Good feedback", "Respected", "Lots to read", "Participation matters",
    "Don’t skip class or you will not pass", "Lots of homework", "Inspirational", "Pop quizzes!",
    "Accessible", "So many papers", "Clear grading", "Hilarious", "Test heavy",
    "Graded by few things", "Amazing lectures", "Caring", "Extra credit", "Group projects",
    "Lecture heavy"
]
df = pd.DataFrame({
    'Index': index_names_tag,        # Column names or indices
    'P-value': p_value_Q4,            # P-values   
    'U-statistics': statistics_Q4,
    'effect size': effect_size_Q4   
})
df = df.sort_values(by='P-value', ascending=True)
# Plot P-value
plt.figure(figsize=(10, 6))
sns.barplot(x='Index', y='P-value', data=df, palette='Blues_d')
plt.title('P-value for Each Index')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot U-statistics
plt.figure(figsize=(10, 6))
sns.barplot(x='Index', y='U-statistics', data=df, palette='Greens_d')
plt.title('U-statistics for Each Index')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot Effect Size
plt.figure(figsize=(10, 6))
sns.barplot(x='Index', y='effect size', data=df, palette='Reds_d')
plt.title('Effect Size for Each Index')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Question 5
# Apply the same data-prepoccessing as Q1
difficulty_gender = rmp_num[[2,3,7,8]][(rmp_num[7] == 1) ^ (rmp_num[8] == 1)]
difficulty_male = difficulty_gender[rating_gender[7]==1]
difficulty_female = difficulty_gender[rating_gender[8]==1]
# transform average difficulty to weighted data according to the # of rating
difficulty_male_weighted = weighted_data(difficulty_male[2], difficulty_male[3])
difficulty_female_weighted = weighted_data(difficulty_female[2], difficulty_female[3])
# Distribution plot
plt.figure(figsize=(8, 6))
sns.kdeplot(difficulty_male_weighted, shade=True,label='Male difficulty', color='blue')
sns.kdeplot(difficulty_female_weighted, shade=True,label='Female difficulty', color='orange')
plt.title('KDE of weighted average difficulty Plot')
plt.xlabel('average difficulty Value')
plt.ylabel('Density')
plt.legend()
plt.show()
# Skewness Check
Skewness_check(difficulty_male_weighted)
Skewness_check(difficulty_female_weighted)
# Check equal variance
stat, p = f_test(difficulty_male_weighted, difficulty_female_weighted)
print(f"F Test: f={stat}, p={p}")
# Apply Weltch t-test to check difference
stat, p = stats.ttest_ind(difficulty_male_weighted, difficulty_female_weighted,equal_var= False)
print(f"t Test: t={stat}, p={p}")
# Cohen's d
print("The cohen's d is ",cohens_d(difficulty_male_weighted, difficulty_female_weighted))
# distribution comparison
stat, p = stats.ks_2samp(difficulty_male_weighted, difficulty_female_weighted)
print(f"KS Statistic: {stat}, p-value: {p}")
if p > 0.05:
    print("The two distributions are likely the same.")
else:
    print("The two distributions are likely different.")
print()

# Question 6
# Apply normal distribution to compute CI according to the CLT
mean1, mean2 = np.mean(difficulty_male_weighted), np.mean(difficulty_female_weighted)
std1, std2 = np.std(difficulty_male_weighted, ddof=1), np.std(difficulty_female_weighted, ddof=1)
n1, n2 = len(difficulty_male_weighted), len(difficulty_female_weighted)
# Standard error
se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
# Critical z-value for 95% confidence
z_crit = stats.norm.ppf(0.975)
# Confidence interval
margin_of_error = z_crit * se
diff_means = mean1 - mean2
lower_bound = diff_means - margin_of_error
upper_bound = diff_means + margin_of_error
print(f"Difference in Means: {diff_means}")
print(f"95% Confidence Interval: [{lower_bound}, {upper_bound}]")
# CI plot
x = np.linspace(diff_means - 4*se, diff_means + 4*se, 1000)
y = stats.norm.pdf(x, loc=diff_means, scale=se)
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Normal Distribution", color="blue")
ci_x = np.linspace(lower_bound, upper_bound, 1000)
ci_y = stats.norm.pdf(ci_x, loc=diff_means, scale=se)
plt.axvline(lower_bound, color="red", linestyle="--", label=f"Lower Bound: {lower_bound}")
plt.axvline(upper_bound, color="red", linestyle="--", label=f"Upper Bound: {upper_bound}")
plt.axvline(diff_means, color="black", linestyle="-", label=f"Mean Difference: {diff_means}")
plt.title("Confidence Interval for Difference in Means (z-distribution)")
plt.xlabel("Difference in Means")
plt.ylabel("Probability Density")
plt.legend()
plt.grid()
plt.show()
print()

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,roc_auc_score,roc_curve,confusion_matrix,ConfusionMatrixDisplay,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from sklearn.linear_model import LogisticRegression
cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_num)
cv1 = KFold(n_splits=5, shuffle=True, random_state=random_num)
def pipeline_Lasso(X,Y,return_test = False):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = random_num)
    # Scale the feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Grid search best alpha
    param_grid = {'alpha': np.linspace(1e-5,1e-1,1000)}
    grid_search = GridSearchCV(Lasso(),param_grid,cv = cv1,scoring = 'neg_mean_squared_error',n_jobs=-1)
    grid_search.fit(X_train_scaled, Y_train , sample_weight = rmp_num[3].loc[X_train.index])
    best_lambda = grid_search.best_params_['alpha']
    print(f"The Best lambda for Lasso is : {best_lambda}")
    # Fit the model
    model = grid_search.best_estimator_
    Y_train_pred = model.predict(X_train_scaled)
    r2_train = r2_score(Y_train, Y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
    Y_test_pred = model.predict(X_test_scaled)
    r2_test = r2_score(Y_test, Y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
    # PLOT Y_pred vs Y for test set
    mask = rmp_num[3].loc[Y_test.index] < 5
    plt.figure(figsize=(8, 6))
    plt.scatter(
        Y_test[mask], Y_test_pred[mask], 
        color='red', alpha=0.6, label='rating # < 5')
    plt.scatter(
        Y_test[~mask], Y_test_pred[~mask], 
        color='blue', alpha=0.6, label='rating # > 5'
    )
    plt.plot(
        [Y_train.min(), Y_train.max()], 
        [Y_train.min(), Y_train.max()], 
        color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values (Goodness of Fit) for test set')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Equation printing
    intercept = model.intercept_
    coefficients = model.coef_
    coefficients_adjusted = coefficients / scaler.scale_
    intercept_adjusted = intercept - np.sum(scaler.mean_ * coefficients/scaler.scale_)
    equation = f"y = {intercept_adjusted :.4f}"
    for feature, coef in zip(X_train.columns, coefficients_adjusted):
        equation += f" + ({coef:.4f}) * {feature}"
    print("Lasso Model Equation:")
    print(equation)
    model.intercept_ = intercept_adjusted
    model.coef_ = coefficients_adjusted
    if return_test == False:
        return model,r2_train,rmse_train,r2_test,rmse_test
    else:
        return model,r2_train,rmse_train,r2_test,rmse_test,X_test,Y_test
def linearity_map(X):
    # Dectect collinearity
    X.columns = [f"X{i}" for i in range(1, len(X.columns) + 1)]
    correlation_matrix = X.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

# Question 7
# row-wise cleaning of rmp_num, drop the row with undetermined gender and drop the last column to avoid dummy trouble 
X7 = row_wise(rmp_num[(rmp_num[7] == 1) ^ (rmp_num[8] == 1)]).drop(columns = [8])
Y7 = X7[1]
X7 = X7.drop(columns=[1])
# Dectect collinearity
X7.columns = [f"X{i}" for i in X7.columns]
correlation_matrix = X7.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, cmap='coolwarm',annot=True, fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
# Model construction
model7,r2_train7,rmse_train7,r2_test7,rmse_test7,X_test7,Y_test7 = pipeline_Lasso(X7,Y7,True)
print(f"RMSE for trainning set: {rmse_train7}")
print(f"R² for the trainning set: {r2_train7}")
print(f"RMSE for the test set: {rmse_test7}")
print(f"R² for the test set: {r2_test7}")
print()

# Question 8
# Define data martrix X and Y
Y8 = rmp_num[1].dropna()
# Normalize the tags data for uniform scale and better interpertation
X8 = rmp_tags.loc[Y8.index].div(rmp_num[3].loc[Y8.index], axis=0)
# Collinearity Check
linearity_map(X8)
# Model construction
model8,r2_train8,rmse_train8,r2_test8,rmse_test8 = pipeline_Lasso(X8,Y8)
print(f"RMSE for trainning set: {rmse_train8}")
print(f"R² for the trainning set: {r2_train8}")
print(f"RMSE for the test set: {rmse_test8}")
print(f"R² for the test set: {r2_test8}")
# Model comparsion
X_corresponds = rmp_tags.loc[X_test7.index].div(rmp_num[3].loc[X_test7.index],axis=0)
Y_test7_predby8 = model8.predict(X_corresponds)
r2_compared = r2_score(Y_test7, Y_test7_predby8)
rmse_compared = np.sqrt(mean_squared_error(Y_test7, Y_test7_predby8))
print(f"R² of the model8 in terms of the test set of Q7 is : {r2_compared}")
print(f"RMSE of the model8 in terms of the test set of Q7 : {rmse_compared}")
print()

# Question 9
# Define predictors matrix X and measurement Y 
Y9 = rmp_num[2].dropna()
# Normalize the tags data for uniform scale and better interpertation
X9 = rmp_tags.loc[Y9.index].div(rmp_num[3].loc[Y9.index], axis=0)
X9.columns = [f"X{i}" for i in range(1, len(X9.columns) + 1)]
#Pipline
# Model construction
model9,r2_train9,rmse_train9,r2_test9,rmse_test9 = pipeline_Lasso(X9,Y9)
print(f"RMSE for trainning set: {rmse_train9}")
print(f"R² for the trainning set: {r2_train9}")
print(f"RMSE for the test set: {rmse_test9}")
print(f"R² for the test set: {r2_test9}")
print() 
# Performance on the data with more rating # 
X_corresponds = rmp_tags.loc[X_test7.index].div(rmp_num[3].loc[X_test7.index],axis=0)
Y_test7_predby9 = model9.predict(X_corresponds)
r2_compared = r2_score(rmp_num[2].loc[Y_test7.index], Y_test7_predby9)
rmse_compared = np.sqrt(mean_squared_error(rmp_num[2].loc[Y_test7.index], Y_test7_predby9))
print(f"R² of the model9 in terms of the test set of Q7 is : {r2_compared}")
print(f"RMSE of the model9 in terms of the test set of Q7 : {rmse_compared}")
print()

# Question 10
# Data preprocessing
rmp_num_clean = row_wise(rmp_num[(rmp_num[7] == 1) ^ (rmp_num[8] == 1)]).drop(columns = [8])
X10 = pd.concat([rmp_num_clean,rmp_tags.loc[rmp_num_clean.index].div(rmp_num[3].loc[rmp_num_clean.index], axis=0)],axis = 1)
num_index = [
    "Average Rating",
    "Average Difficulty",
    "Number of ratings",
    "Received a 'pepper'?",
    "The proportion of students that said they would take the class again",
    "The number of ratings coming from online classes",
    "if male",
]
X10_index = num_index + index_names_tag
X10.columns = X10_index
Y10 = X10["Received a 'pepper'?"]
X10 = X10.drop(columns = ["Received a 'pepper'?"])
# Balance check
print(np.sum(Y10==1),np.sum(Y10==0))
# model construction
X10_train, X10_test, Y10_train, y10_test = train_test_split(X10, Y10, test_size=0.2, random_state= random_num,stratify=Y10)
scaler = StandardScaler()
X10_train_scaled = scaler.fit_transform(X10_train)
X10_test_scaled = scaler.transform(X10_test)
lasso_model = LogisticRegression(penalty='l1', solver='saga',class_weight='balanced', max_iter=5000,random_state= random_num)
param_grid = {'C': np.linspace(1e-2,1,1000)}
grid_search = GridSearchCV(
    estimator=lasso_model,
    param_grid=param_grid,
    scoring= 'roc_auc',
    cv=cv2,
    verbose=1,
    n_jobs=-1,
)
grid_search.fit(X10_train_scaled,Y10_train)
best_model = grid_search.best_estimator_
y10_pred_proba = best_model.predict_proba(X10_test_scaled)[:, 1]
roc_auc = roc_auc_score(y10_test, y10_pred_proba)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Test ROC-AUC Score: {roc_auc}")
fpr, tpr, thresholds = roc_curve(y10_test, y10_pred_proba)
# Plot the ROC curve
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()
# Choose the best threshold
best_threshold_index = np.argmax(tpr - fpr)
best_threshold = thresholds[best_threshold_index]
print(f"Best Threshold: {best_threshold:.2f}")
# Confusion matrix
y10_pred = (y10_pred_proba >= best_threshold).astype(int)
cm = confusion_matrix(y10_test, y10_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
# transform the coefficients back to the original scale
intercept = best_model.intercept_
coefficients = best_model.coef_
coefficients_adjusted = coefficients / scaler.scale_
intercept_adjusted = intercept - np.sum(scaler.mean_ * coefficients/scaler.scale_)
best_model.intercept_ = intercept_adjusted
best_model.coef_ = coefficients_adjusted
# Print intercept
print(f"Intercept: {best_model.intercept_}")
df = pd.DataFrame({
    'Index': X10.columns.values,
    'Coeff':best_model.coef_[0]
}) 
# Plot coeff barplots
plt.figure(figsize=(10, 6))
sns.barplot(x='Index', y='Coeff', data=df, palette='Greens_d')
plt.title('Coeff for Each Index')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
















