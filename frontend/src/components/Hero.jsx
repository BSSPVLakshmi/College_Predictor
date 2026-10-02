export default function Hero() {
    return (
        <section className="hero">
            {/* Logo on left */}
            <div className="logo">
                <img src="/logo.png" alt="College Predictor Logo" />
            </div>

            {/* Center text */}
            <div className="hero-text">
                <h1 className="hero-title">
                    College Predictor – Find Your Best Engineering College
                </h1>

                <p className="hero-sub">
                    Predict colleges based on your rank, category, branch, and location.
                </p>
            </div>
        </section>
    );
}
