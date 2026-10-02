import { useState } from "react";

const slides = [
    {
        color: "blue",
        img: "/feature-slide-1.png",
        title: "Smart College Prediction",
        text:
            "Predict engineering colleges based on your EAPCET rank, category, gender, branch, and district preferences."
    },
    {
        color: "green",
        img: "/feature-slide-2.png",
        title: "Personalized Filtering",
        text:
            "Apply multiple filters like branch and district to shortlist colleges that best match your interests."
    },
    {
        color: "peach",
        img: "/feature-slide-3.png",
        title: "Fast & Accurate Results",
        text:
            "Get quick, clear, and structured results to make confident admission decisions."
    }
];

export default function Features() {
    const [active, setActive] = useState(0);

    const nextSlide = () => {
        setActive((prev) => (prev + 1) % slides.length);
    };

    return (
        <section className="features-slider">

            {/* HEADING */}
            <div className="features-header">
                <span className="features-tag">EAPCET Predictor</span>
                <h2>Engineering College Predictor – Discover Your Best Options</h2>
            </div>

            {/* SLIDER */}
            <div className="slider-window">
                <div
                    className="slider-track"
                    style={{ transform: `translateX(-${active * 100}%)` }}
                >
                    {slides.map((s, i) => (
                        <div
                            key={i}
                            className={`slide ${s.color}`}
                            onClick={nextSlide}   /* 👈 CLICK BOX TO CHANGE */
                        >
                            <div className="slide-img">
                                <img src={s.img} alt={s.title} />
                            </div>
                            <div className="slide-text">
                                <h3>{s.title}</h3>
                                <p>{s.text}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* DOTS */}
            <div className="slider-dots">
                {slides.map((_, i) => (
                    <span
                        key={i}
                        className={i === active ? "dot active" : "dot"}
                        onClick={() => setActive(i)}
                    />
                ))}
            </div>

        </section>
    );
}
