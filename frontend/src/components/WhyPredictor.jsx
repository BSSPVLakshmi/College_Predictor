export default function WhyPredictor() {
    return (
        <section className="why-section">
            <div className="why-card">

                {/* LEFT CONTENT */}
                <div className="why-text">
                    <h2>Engineering College Predictor </h2>

                    <h3>What is an Engineering College Predictor?</h3>
                    <p>
                        An Engineering College Predictor is a smart tool designed to help
                        students estimate which engineering colleges they may get admission
                        to based on their EAPCET rank. By analyzing previous year cut-offs,
                        category-wise reservations, gender-based seat allocation, branch
                        availability, and district preferences, the system provides a
                        reliable list of eligible colleges.
                    </p>

                    <h3>How Does the Engineering College Predictor Work?</h3>
                    <ul>
                        <li>
                            <strong>Input Your Details:</strong> Enter your EAPCET rank,
                            category, gender, preferred branch, and district or region.
                        </li>
                        <li>
                            <strong>Data Analysis & Prediction:</strong> The system compares
                            your rank with historical cut-off data while applying official
                            admission rules and reservation policies.
                        </li>
                        <li>
                            <strong>College Recommendations:</strong> Generates a list of
                            engineering colleges where you have a high probability of
                            securing admission.
                        </li>
                    </ul>

                    <h3>Key Features</h3>
                    <ul>
                        <li>Accurate predictions based on previous EAPCET cut-off trends</li>
                        <li>Category and gender-specific college recommendations</li>
                        <li>Branch-wise and district-wise filtering</li>
                        <li>Supports government and private engineering colleges</li>
                        <li>Simple and user-friendly interface</li>
                    </ul>

                    <h3>Why Use This College Predictor?</h3>
                    <ul>
                        <li>Reduces uncertainty during college selection</li>
                        <li>Saves time by avoiding manual cut-off comparisons</li>
                        <li>Helps shortlist colleges based on rank and preference</li>
                        <li>Assists students in making informed counseling decisions</li>
                    </ul>
                </div>

                {/* RIGHT IMAGE */}
                <div className="why-image">
                    <img
                        src="/images/engineering-predictor.png"
                        alt="Engineering College Predictor Illustration"
                    />
                </div>

            </div>
        </section>
    );
}
