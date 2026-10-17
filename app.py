import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="StudentIQ | Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. DARK THEME STYLING
# ==========================================================

st.markdown("""
<style>

    /* Main application */

    .stApp {
        background-color: #0B1120;
        color: #E5E7EB;
    }

    [data-testid="stHeader"] {
        background-color: #0B1120;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* Sidebar */

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263449;
    }

    [data-testid="stSidebar"] * {
        color: #D1D5DB;
    }

    /* Main title */

    .brand-title {
        font-size: 34px;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -1px;
    }

    .brand-accent {
        color: #38BDF8;
    }

    .page-subtitle {
        color: #94A3B8;
        font-size: 15px;
        margin-top: 5px;
        margin-bottom: 20px;
    }

    /* Section headings */

    h1, h2, h3 {
        color: #F1F5F9 !important;
    }

    p, label {
        color: #CBD5E1;
    }

    /* Metric cards */

    [data-testid="stMetric"] {
        background-color: #151F32;
        border: 1px solid #26364F;
        border-radius: 12px;
        padding: 20px;
        min-height: 120px;
    }

    [data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-weight: 700;
    }

    [data-testid="stMetricDelta"] {
        color: #34D399 !important;
    }

    /* Cards */

    .panel {
        background-color: #151F32;
        border: 1px solid #26364F;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .panel-title {
        color: #F8FAFC;
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .muted-text {
        color: #94A3B8;
        font-size: 13px;
    }

    /* Buttons */

    .stButton > button {
        background-color: #2563EB;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 9px 18px;
        font-weight: 600;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
        border: 1px solid #60A5FA;
    }

    .stDownloadButton > button {
        background-color: #172554;
        color: #BFDBFE;
        border: 1px solid #3B82F6;
        border-radius: 8px;
    }

    /* Inputs */

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background-color: #111827;
        border-color: #334155;
    }

    input {
        color: #F8FAFC !important;
    }

    /* Dataframes */

    [data-testid="stDataFrame"] {
        border: 1px solid #263449;
        border-radius: 10px;
    }

    /* Divider */

    hr {
        border-color: #263449;
    }

    /* Footer */

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 12px;
        padding-top: 20px;
        border-top: 1px solid #263449;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. PLOT THEME
# ==========================================================

sns.set_theme(
    style="darkgrid",
    rc={
        "figure.facecolor": "#151F32",
        "axes.facecolor": "#151F32",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#CBD5E1",
        "text.color": "#F1F5F9",
        "xtick.color": "#CBD5E1",
        "ytick.color": "#CBD5E1",
        "grid.color": "#263449",
        "grid.alpha": 0.5
    }
)


# ==========================================================
# 4. LOAD DATASET
# ==========================================================

DATA_FILE = Path("Student_performance_data _.csv")


@st.cache_data
def load_data(file_path, modified_time):

    data = pd.read_csv(file_path)

    return data


if not DATA_FILE.exists():

    st.error(
        "Dataset not found. Place "
        "'Student_performance_data _.csv' "
        "in the same folder as app.py."
    )

    st.stop()


df = load_data(
    str(DATA_FILE),
    DATA_FILE.stat().st_mtime
)


# ==========================================================
# 5. VALIDATE DATASET
# ==========================================================

required_columns = [
    "StudentID",
    "Age",
    "Gender",
    "Ethnicity",
    "ParentalEducation",
    "StudyTimeWeekly",
    "Absences",
    "Tutoring",
    "ParentalSupport",
    "Extracurricular",
    "Sports",
    "Music",
    "Volunteering",
    "GPA",
    "GradeClass"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


if df.empty:

    st.error("The dataset is empty.")

    st.stop()


df["GradeClass"] = df["GradeClass"].astype(int)


# ==========================================================
# 6. HEADER
# ==========================================================

header_left, header_right = st.columns([3, 1])

with header_left:

    st.markdown(
        """
        <div class="brand-title">
            Student<span class="brand-accent">IQ</span>
        </div>
        <div class="page-subtitle">
            STUDENT PERFORMANCE INTELLIGENCE PLATFORM
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:

    st.markdown(
        """
        <div style="text-align:right; padding-top:10px;">
            <span style="
                background:#12352F;
                color:#34D399;
                padding:8px 12px;
                border-radius:20px;
                font-size:12px;
                border:1px solid #166534;">
                ● SYSTEM ONLINE
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# ==========================================================
# 7. SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.markdown(
    """
    <div style="padding:10px 0 20px 0;">
        <div style="
            font-size:25px;
            font-weight:800;
            color:#F8FAFC;">
            Student<span style="color:#38BDF8;">IQ</span>
        </div>
        <div style="
            font-size:11px;
            color:#64748B;
            letter-spacing:1px;">
            ANALYTICS WORKSPACE
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "WORKSPACE",
    [
        "Overview",
        "Academic Analytics",
        "Model Performance",
        "Student Prediction"
    ]
)

st.sidebar.divider()


st.sidebar.caption(
    "StudentIQ "
)


# ==========================================================
# 8. MODEL TRAINING FUNCTION
# ==========================================================

feature_columns = [
    col for col in df.columns
    if col not in [
        "StudentID",
        "GPA",
        "GradeClass"
    ]
]


@st.cache_resource
def train_models(data):

    features = data[feature_columns].copy()

    target = data["GradeClass"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target
    )

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )

    }

    trained_models = {}
    predictions = {}
    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        predicted = model.predict(X_test)

        trained_models[name] = model

        predictions[name] = predicted

        results[name] = {
            "Accuracy": accuracy_score(
                y_test, predicted
            ),
            "Precision": precision_score(
                y_test,
                predicted,
                average="weighted",
                zero_division=0
            ),
            "Recall": recall_score(
                y_test,
                predicted,
                average="weighted",
                zero_division=0
            ),
            "F1 Score": f1_score(
                y_test,
                predicted,
                average="weighted",
                zero_division=0
            )
        }

    results_df = pd.DataFrame(results).T

    results_df.index.name = "Model"

    return (
        trained_models,
        results_df,
        X_test,
        y_test,
        predictions
    )


# ==========================================================
# 9. REUSABLE CHART SETTINGS
# ==========================================================

def style_axis(ax):

    ax.set_facecolor("#151F32")

    ax.tick_params(colors="#CBD5E1")

    ax.xaxis.label.set_color("#CBD5E1")

    ax.yaxis.label.set_color("#CBD5E1")

    ax.title.set_color("#F8FAFC")

    ax.grid(
        color="#334155",
        alpha=0.4
    )


# ==========================================================
# 10. OVERVIEW PAGE
# ==========================================================

if page == "Overview":

    st.subheader("Executive Overview")

    st.caption(
        "A high-level summary of student academic data."
    )

    # KPI cards

    total_students = len(df)

    average_gpa = df["GPA"].mean()

    average_study = df["StudyTimeWeekly"].mean()

    total_absences = df["Absences"].sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Students",
        f"{total_students:,}"
    )

    c2.metric(
        "Average GPA",
        f"{average_gpa:.2f}"
    )

    c3.metric(
        "Weekly Study Hours",
        f"{average_study:.1f}"
    )

    c4.metric(
        "Total Absences",
        f"{total_absences:,.0f}"
    )

    st.write("")

    # Charts

    left, right = st.columns(2)

    with left:

        st.markdown("### Performance Distribution")

        counts = (
            df["GradeClass"]
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        bars = ax.bar(
            counts.index.astype(str),
            counts.values,
            color="#38BDF8",
            width=0.6
        )

        ax.set_xlabel("Performance Class")

        ax.set_ylabel("Number of Students")

        ax.set_title("Students by Performance Class")

        ax.bar_label(
            bars,
            color="#F8FAFC",
            padding=3
        )

        style_axis(ax)

        st.pyplot(fig)

        plt.close(fig)

    with right:

        st.markdown("### Study Time Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(
            df["StudyTimeWeekly"],
            bins=20,
            kde=True,
            color="#818CF8",
            ax=ax
        )

        ax.set_xlabel("Weekly Study Hours")

        ax.set_ylabel("Number of Students")

        ax.set_title("Distribution of Study Time")

        style_axis(ax)

        st.pyplot(fig)

        plt.close(fig)

    st.divider()

    # Dataset preview

    st.markdown("### Student Records")

    search_id = st.text_input(
        "Search Student ID",
        placeholder="Enter a student ID or leave blank"
    )

    display_df = df.copy()

    if search_id.strip():

        display_df = display_df[
            display_df["StudentID"].astype(str)
            .str.contains(search_id.strip(), case=False)
        ]

    st.dataframe(
        display_df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_performance_data.csv",
        mime="text/csv"
    )


# ==========================================================
# 11. ACADEMIC ANALYTICS PAGE
# ==========================================================

elif page == "Academic Analytics":

    st.subheader("Academic Analytics")

    st.caption(
        "Explore academic patterns and relationships "
        "between student attributes."
    )

    # Filters

    f1, f2 = st.columns(2)

    with f1:

        selected_classes = st.multiselect(
            "Select Performance Classes",
            options=sorted(df["GradeClass"].unique()),
            default=sorted(df["GradeClass"].unique())
        )

    with f2:

        max_absences = st.slider(
            "Maximum Absences",
            min_value=0,
            max_value=max(1, int(df["Absences"].max())),
            value=int(df["Absences"].max())
        )

    filtered_df = df[
        df["GradeClass"].isin(selected_classes)
        & (df["Absences"] <= max_absences)
    ]

    st.write(
        f"Records matching filters: **{len(filtered_df):,}**"
    )

    st.divider()

    # Study time and GPA

    left, right = st.columns(2)

    with left:

        st.markdown("### Study Time vs GPA")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.scatterplot(
            data=filtered_df,
            x="StudyTimeWeekly",
            y="GPA",
            hue="GradeClass",
            palette="cool",
            alpha=0.75,
            ax=ax
        )

        ax.set_title("Study Hours and GPA")

        style_axis(ax)

        ax.legend(
            title="Class",
            facecolor="#151F32",
            labelcolor="#E5E7EB"
        )

        st.pyplot(fig)

        plt.close(fig)

    with right:

        st.markdown("### Absences vs GPA")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.scatterplot(
            data=filtered_df,
            x="Absences",
            y="GPA",
            hue="GradeClass",
            palette="cool",
            alpha=0.75,
            ax=ax
        )

        ax.set_title("Absences and GPA")

        style_axis(ax)

        ax.legend(
            title="Class",
            facecolor="#151F32",
            labelcolor="#E5E7EB"
        )

        st.pyplot(fig)

        plt.close(fig)

    st.divider()

    # Tutoring comparison

    st.markdown("### Average GPA by Tutoring Status")

    tutoring_data = (
        filtered_df.groupby("Tutoring")["GPA"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=tutoring_data,
        x="Tutoring",
        y="GPA",
        color="#38BDF8",
        ax=ax
    )

    ax.set_xticks([0, 1])

    ax.set_xticklabels([
        "No Tutoring (0)",
        "Tutoring (1)"
    ])

    ax.set_title("Average GPA by Tutoring")

    style_axis(ax)

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # Correlation heatmap

    st.markdown("### Feature Correlation Matrix")

    numeric_df = filtered_df.select_dtypes(
        include=np.number
    ).drop(
        columns=["StudentID"],
        errors="ignore"
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.3,
        ax=ax
    )

    ax.set_title("Correlation Between Numerical Variables")

    st.pyplot(fig)

    plt.close(fig)

    st.caption(
        "Correlation indicates association, not necessarily causation."
    )


# ==========================================================
# 12. MODEL PERFORMANCE PAGE
# ==========================================================

elif page == "Model Performance":

    st.subheader("Model Performance")

    st.caption(
        "Compare machine learning algorithms using "
        "classification evaluation metrics."
    )

    with st.spinner(
        "Training models and calculating metrics..."
    ):

        (
            trained_models,
            results_df,
            X_test,
            y_test,
            predictions
        ) = train_models(df)

    st.success("Model evaluation completed.")

    # Model comparison

    st.markdown("### Model Comparison")

    formatted_df = results_df.copy()

    for col in formatted_df.columns:

        formatted_df[col] = (
            formatted_df[col] * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        formatted_df,
        use_container_width=True
    )

    st.divider()

    # Metrics chart

    st.markdown("### Evaluation Metrics")

    metric = st.selectbox(
        "Select Metric",
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    )

    fig, ax = plt.subplots(figsize=(9, 4))

    values = results_df[metric] * 100

    bars = ax.bar(
        results_df.index,
        values,
        color=["#38BDF8", "#818CF8", "#34D399"]
    )

    ax.set_ylabel(f"{metric} (%)")

    ax.set_title(f"{metric} Comparison")

    ax.set_ylim(0, 100)

    ax.bar_label(
        bars,
        fmt="%.2f%%",
        color="#F8FAFC",
        padding=3
    )

    style_axis(ax)

    plt.xticks(rotation=10)

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # Best model

    best_model_name = results_df["F1 Score"].idxmax()

    best_model = trained_models[best_model_name]

    col1, col2 = st.columns(2)

    col1.metric(
        "Selected Model",
        best_model_name
    )

    col2.metric(
        "Weighted F1 Score",
        f"{results_df.loc[best_model_name, 'F1 Score']:.2%}"
    )

    st.info(
        "The displayed model is selected by weighted F1-score "
        "on the test split. For a rigorous final evaluation, "
        "select the model using validation data or cross-validation."
    )

    st.divider()

    # Confusion matrix

    st.markdown("### Confusion Matrix")

    y_pred = predictions[best_model_name]

    labels = sorted(df["GradeClass"].unique())

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )

    ax.set_xlabel("Predicted Class")

    ax.set_ylabel("Actual Class")

    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # Classification report

    st.markdown("### Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    # Save model

    if st.button("Save Selected Model"):

        joblib.dump(
            best_model,
            "student_performance_model.pkl"
        )

        st.success(
            f"{best_model_name} saved successfully."
        )


# ==========================================================
# 13. STUDENT PREDICTION PAGE
# ==========================================================

elif page == "Student Prediction":

    st.subheader("Student Performance Prediction")

    st.caption(
        "Enter student information to estimate "
        "the performance class."
    )

    with st.spinner("Preparing the prediction model..."):

        (
            trained_models,
            results_df,
            X_test,
            y_test,
            predictions
        ) = train_models(df)

    best_model_name = results_df["F1 Score"].idxmax()

    best_model = trained_models[best_model_name]

    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">Prediction Engine</div>
            <div class="muted-text">
                Selected model: {best_model_name}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Input form

    with st.form("prediction_form"):

        st.markdown("### Student Information")

        left, right = st.columns(2)

        input_values = {}

        categorical_features = [
            "Gender",
            "Ethnicity",
            "ParentalEducation",
            "Tutoring",
            "ParentalSupport",
            "Extracurricular",
            "Sports",
            "Music",
            "Volunteering"
        ]

        for index, feature in enumerate(feature_columns):

            series = df[feature]

            with (left if index % 2 == 0 else right):

                if feature in categorical_features:

                    choices = sorted(
                        series.dropna().unique().tolist()
                    )

                    input_values[feature] = st.selectbox(
                        feature,
                        choices,
                        key=f"prediction_{feature}"
                    )

                else:

                    minimum = float(series.min())

                    maximum = float(series.max())

                    default = float(series.median())

                    if pd.api.types.is_integer_dtype(series):

                        input_values[feature] = st.number_input(
                            feature,
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(default),
                            step=1,
                            key=f"prediction_{feature}"
                        )

                    else:

                        input_values[feature] = st.number_input(
                            feature,
                            min_value=minimum,
                            max_value=maximum,
                            value=default,
                            step=0.5,
                            key=f"prediction_{feature}"
                        )

        submitted = st.form_submit_button(
            "Predict Performance"
        )

    if submitted:

        new_student = pd.DataFrame(
            [input_values],
            columns=feature_columns
        )

        try:

            predicted_class = best_model.predict(
                new_student
            )[0]

            st.divider()

            st.markdown("### Prediction Result")

            result_col1, result_col2 = st.columns(2)

            result_col1.metric(
                "Predicted Class",
                str(predicted_class)
            )

            result_col2.metric(
                "Model Used",
                best_model_name
            )

            st.success(
                f"Predicted performance class: {predicted_class}"
            )

            # Class probabilities

            if hasattr(best_model, "predict_proba"):

                probabilities = best_model.predict_proba(
                    new_student
                )[0]

                classes = best_model.classes_

                probability_df = pd.DataFrame({
                    "Class": classes,
                    "Estimated Probability": probabilities
                })

                st.markdown("### Class Probability Estimates")

                st.bar_chart(
                    probability_df.set_index("Class"),
                    y="Estimated Probability"
                )

            # Download result

            result = new_student.copy()

            result["PredictedGradeClass"] = predicted_class

            st.download_button(
                "Download Prediction",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="student_prediction.csv",
                mime="text/csv"
            )

            st.caption(
                "This prediction is an estimate based on "
                "historical data. It is not a definitive "
                "judgment of student ability."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ==========================================================
# 14. FOOTER
# ==========================================================

st.markdown(
    """
    <div class="footer">
        StudentIQ Analytics &nbsp; | &nbsp;
    </div>
    """,
    unsafe_allow_html=True
)