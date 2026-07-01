TOOL_BIRTH_YEARS = {
    "PyTorch": 2016,
    "FAISS": 2017,
    "transformers": 2018,
    "LangChain": 2022,
    "GPT-4": 2023,
}

COMPANY_FOUNDING_YEARS = {
    "Google": 1998,
    "Meta": 2004,
    "Amazon": 1994,
    "Microsoft": 1975,
    "Flipkart": 2007,
    "Swiggy": 2014,
    "Razorpay": 2010,
    "Freshworks": 2010,
    "Infosys": 1981,
    "TCS": 1968,
    "Wipro": 1945,
    "Accenture": 1989,
    "OpenAI": 2015,
    "Anthropic": 2021,
    "HuggingFace": 2016,
    "StartupX": 2020,
    "TechCorp": 2018,
    "DataLabs": 2019,
    "MLWorks": 2017,
    "SearchCo": 2016,
}

ESCO_SKILL_GROUPS = {
    "Deep Learning": {"JAX", "PyTorch", "TensorFlow", "Keras", "deep learning"},
    "Search Infrastructure": {"Elasticsearch", "OpenSearch", "Solr", "FAISS", "BM25"},
    "GBDT": {"LightGBM", "XGBoost", "CatBoost", "gradient boosting"},
    "NLP": {"BERT", "transformers", "spaCy", "NLTK", "NLP"},
    "MLOps": {"MLflow", "Kubeflow", "Docker", "Kubernetes", "CI/CD"},
    "Retrieval": {"BGE", "dense retrieval", "vector search", "embedding", "NDCG"},
}

OPENING_VARIANTS = [
    "Strongest signal is {signal} — {detail}.",
    "Career arc shows {trajectory}; currently {title} at {employer}.",
    "{years}yr track record in {domain} with {achievement}.",
    "Shipped {product} at {employer}, directly matching the JD's {jd_requirement}.",
    "Stands out for {skill} and {skill2}, both explicitly required by the role.",
    "{employer} background ({employer_type}) aligns with the product-co preference.",
    "Active in the last {last_active} days; {response_rate}% recruiter response rate signals high availability.",
    "Eval expertise ({eval_skills}) covers the JD's retrieval quality focus directly.",
]

CONCERN_VARIANTS = {
    "notice_period": [
        "One friction point: {notice_period}-day notice period against the ≤30-day preference.",
        "Timeline risk: {notice_period}-day notice extends past the target onboarding window.",
    ],
    "services_background": [
        "Consulting-heavy history ({services_ratio}% of career) — product shipping velocity may need calibration.",
        "Services background at {firm} will require adjustment to the product cadence expected here.",
    ],
    "no_code": [
        "Transition to {current_title} role suggests reduced hands-on coding in last {months}mo — IC ramp expected.",
        "Architecture-focused last {months} months; coding intensity at IC level will need reconfirmation.",
    ],
    "pure_research": [
        "Academic trajectory ({research_ratio}% research roles) — production deployment bias needs demonstration.",
        "Research pedigree is strong; JD's bias toward shipped systems is the key question to probe.",
    ],
    "no_concern": [
        "No gaps identified; growth edge would be scaling the system to {scale}.",
        "Strong fit across all dimensions; L2R depth is the one area not explicitly evidenced.",
        "Near-complete match — only question is team culture fit at this seniority level.",
    ],
}
