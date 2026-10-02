import { useEffect, useRef, useState } from "react";

function MultiSelectDropdown({
    label,
    icon,
    options,
    selected,
    setSelected,
    placeholder,
    accent = "#2563eb",
}) {
    const [open, setOpen] = useState(false);
    const boxRef = useRef(null);

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (boxRef.current && !boxRef.current.contains(event.target)) {
                setOpen(false);
            }
        };

        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    const toggleOption = (value) => {
        if (selected.includes(value)) {
            setSelected(selected.filter((item) => item !== value));
        } else {
            setSelected([...selected, value]);
        }
    };

    const removeItem = (value) => {
        setSelected(selected.filter((item) => item !== value));
    };

    const styles = {
        wrapper: {
            position: "relative",
            width: "100%",
        },
        labelRow: {
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "space-between",
            marginBottom: "8px",
        },
        labelLeft: {
            display: "flex",
            alignItems: "center",
            gap: "8px",
            fontWeight: 700,
            color: "#111827",
        },
        countBadge: {
            fontSize: "12px",
            padding: "4px 8px",
            borderRadius: "999px",
            background: "#eff6ff",
            color: accent,
            border: `1px solid ${accent}22`,
            fontWeight: 700,
        },
        box: {
            minHeight: "48px",
            border: "1px solid #d1d5db",
            borderRadius: "16px",
            padding: "10px 12px",
            background: "#fff",
            cursor: "pointer",
            display: "flex",
            flexWrap: "wrap",
            gap: "8px",
            alignItems: "center",
            transition: "border-color 0.2s ease, box-shadow 0.2s ease",
            boxShadow: open ? `0 0 0 3px ${accent}18` : "none",
            borderColor: open ? accent : "#d1d5db",
        },
        placeholder: {
            color: "#9ca3af",
            display: "flex",
            alignItems: "center",
            gap: "8px",
        },
        tag: {
            background: "linear-gradient(135deg, #dbeafe, #eff6ff)",
            color: "#1d4ed8",
            padding: "6px 10px",
            borderRadius: "999px",
            fontSize: "14px",
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            border: "1px solid #bfdbfe",
            boxShadow: "0 4px 10px rgba(37, 99, 235, 0.08)",
        },
        tagIcon: {
            width: "18px",
            height: "18px",
            borderRadius: "999px",
            background: "#fff",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "12px",
            flexShrink: 0,
        },
        remove: {
            border: "none",
            background: "transparent",
            color: "#1d4ed8",
            fontWeight: 800,
            cursor: "pointer",
            fontSize: "16px",
            lineHeight: 1,
            padding: 0,
        },
        caret: {
            marginLeft: "auto",
            color: "#6b7280",
            fontSize: "14px",
            fontWeight: 700,
            paddingLeft: "10px",
        },
        dropdown: {
            position: "absolute",
            zIndex: 50,
            marginTop: "10px",
            width: "100%",
            height: "260px",
            overflowY: "scroll",
            overflowX: "hidden",
            overscrollBehavior: "contain",
            border: "1px solid #d1d5db",
            background: "#fff",
            borderRadius: "16px",
            boxShadow: "0 18px 40px rgba(0,0,0,0.14)",
        },
        option: {
            display: "flex",
            alignItems: "center",
            gap: "12px",
            padding: "12px 14px",
            cursor: "pointer",
            transition: "background-color 0.15s ease",
            justifyContent: "space-between",
        },
        optionLeft: {
            display: "flex",
            alignItems: "center",
            gap: "10px",
            minWidth: 0,
        },
        optionIcon: {
            width: "28px",
            height: "28px",
            borderRadius: "999px",
            background: `${accent}14`,
            color: accent,
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "14px",
            flexShrink: 0,
        },
        optionText: {
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            fontSize: "15px",
            fontWeight: 500,
            color: "#111827",
        },
        optionRight: {
            display: "flex",
            alignItems: "center",
            gap: "10px",
            flexShrink: 0,
        },
        checkBox: {
            width: "18px",
            height: "18px",
            accentColor: accent,
            cursor: "pointer",
        },
        checkMark: {
            width: "22px",
            height: "22px",
            borderRadius: "999px",
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            background: "#ecfdf5",
            color: "#16a34a",
            fontWeight: 900,
            fontSize: "14px",
        },
        empty: {
            padding: "12px 14px",
            color: "#6b7280",
        },
    };

    return (
        <div ref={boxRef} style={styles.wrapper}>
            <div style={styles.labelRow}>
                <label style={styles.labelLeft}>
                    <span>{icon}</span>
                    <span>{label}</span>
                </label>
                <span style={styles.countBadge}>{selected.length} selected</span>
            </div>

            <div onClick={() => setOpen(!open)} style={styles.box}>
                {selected.length === 0 ? (
                    <span style={styles.placeholder}>
                        <span>{icon}</span>
                        <span>{placeholder}</span>
                    </span>
                ) : (
                    selected.map((item) => (
                        <span key={item} style={styles.tag}>
                            <span style={styles.tagIcon}>{icon}</span>
                            <span>{item}</span>
                            <button
                                type="button"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    removeItem(item);
                                }}
                                style={styles.remove}
                                aria-label={`Remove ${item}`}
                            >
                                ×
                            </button>
                        </span>
                    ))
                )}

                <span style={styles.caret}>{open ? "▴" : "▾"}</span>
            </div>

            {open && (
                <div style={styles.dropdown}>
                    {options.length > 0 ? (
                        options.map((option) => {
                            const isSelected = selected.includes(option);
                            return (
                                <label
                                    key={option}
                                    style={{
                                        ...styles.option,
                                        backgroundColor: isSelected ? "#f8fafc" : "#fff",
                                        borderLeft: isSelected
                                            ? `4px solid ${accent}`
                                            : "4px solid transparent",
                                    }}
                                    onMouseEnter={(e) => {
                                        if (!isSelected) {
                                            e.currentTarget.style.backgroundColor = "#f9fafb";
                                        }
                                    }}
                                    onMouseLeave={(e) => {
                                        e.currentTarget.style.backgroundColor = isSelected
                                            ? "#f8fafc"
                                            : "#fff";
                                    }}
                                >
                                    <div style={styles.optionLeft}>
                                        <span style={styles.optionIcon}>{icon}</span>
                                        <span style={styles.optionText}>{option}</span>
                                    </div>

                                    <div style={styles.optionRight}>
                                        <input
                                            type="checkbox"
                                            checked={isSelected}
                                            onChange={() => toggleOption(option)}
                                            style={styles.checkBox}
                                        />
                                        <span
                                            style={{
                                                ...styles.checkMark,
                                                opacity: isSelected ? 1 : 0,
                                            }}
                                        >
                                            ✓
                                        </span>
                                    </div>
                                </label>
                            );
                        })
                    ) : (
                        <div style={styles.empty}>No options available</div>
                    )}
                </div>
            )}
        </div>
    );
}

function PredictorSteps() {
    const [rank, setRank] = useState("");
    const [category, setCategory] = useState("");
    const [gender, setGender] = useState("FEMALE");
    const [districts, setDistricts] = useState([]);
    const [branches, setBranches] = useState([]);
    const [sortChoice, setSortChoice] = useState("1");
    const [showOnlySafeTarget, setShowOnlySafeTarget] = useState("no");

    const [districtOptions, setDistrictOptions] = useState([]);
    const [branchOptions, setBranchOptions] = useState([]);
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    useEffect(() => {
        const loadOptions = async () => {
            try {
                const res = await fetch("http://127.0.0.1:5000/options");
                const data = await res.json();
                setDistrictOptions(data.districts || []);
                setBranchOptions(data.branches || []);
            } catch {
                setError("Could not load districts and branches.");
            }
        };

        loadOptions();
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError("");
        setResults([]);

        try {
            const body = {
                rank: Number(rank),
                category: category.trim().toUpperCase(),
                gender: gender.trim().toUpperCase(),
                districts: districts.length > 0 ? districts : "ALL",
                branches: branches.length > 0 ? branches : "ALL",
                sort_choice: sortChoice,
                show_only_safe_target: showOnlySafeTarget,
            };

            const res = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(body),
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || "Something went wrong");
            }

            setResults(data.results || []);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const pageStyle = {
        minHeight: "100vh",
        background: "linear-gradient(180deg, #eef4ff 0%, #f7f7fb 100%)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: "12px",
    };

    const shellStyle = {
        width: "100%",
        maxWidth: "1180px",
    };

    const titleStyle = {
        textAlign: "center",
        fontSize: "42px",
        fontWeight: 800,
        marginBottom: "34px",
        color: "#1d4ed8",
        letterSpacing: "-0.02em",
    };

    const formStyle = {
        width: "100%",
        background: "rgba(255,255,255,0.92)",
        backdropFilter: "blur(10px)",
        borderRadius: "28px",
        boxShadow: "0 24px 60px rgba(15, 23, 42, 0.12)",
        padding: "34px",
        display: "grid",
        gap: "22px",
        gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
        border: "1px solid rgba(255,255,255,0.65)",
    };

    const fieldStyle = {
        display: "block",
        marginBottom: "8px",
        fontWeight: 700,
        color: "#111827",
    };

    const inputStyle = {
        width: "100%",
        padding: "12px 14px",
        border: "1px solid #d1d5db",
        borderRadius: "14px",
        fontSize: "16px",
        background: "#fff",
        boxSizing: "border-box",
        outline: "none",
    };

    const buttonStyle = {
        width: "100%",
        background: "linear-gradient(135deg, #2563eb, #1d4ed8)",
        color: "#fff",
        border: "none",
        borderRadius: "14px",
        padding: "13px 18px",
        fontSize: "16px",
        fontWeight: 800,
        cursor: "pointer",
        boxShadow: "0 14px 28px rgba(37,99,235,0.24)",
    };

    const errorStyle = {
        marginTop: "24px",
        padding: "14px",
        background: "#fee2e2",
        color: "#b91c1c",
        borderRadius: "14px",
        border: "1px solid #fecaca",
    };

    const tableWrapStyle = {
        marginTop: "24px",
        overflowX: "auto",
        background: "rgba(255,255,255,0.92)",
        borderRadius: "28px",
        boxShadow: "0 20px 45px rgba(0,0,0,0.12)",
        border: "1px solid rgba(255,255,255,0.4)",
        backdropFilter: "blur(10px)",
    };

    const tableStyle = {
        width: "100%",
        borderCollapse: "collapse",
    };

    const thTdStyle = {
        borderBottom: "1px solid #dbe7e3",
        padding: "14px 16px",
        textAlign: "left",
        fontSize: "15px",
        color: "#164e63",
    };

    const tableHeaderStyle = {
        borderBottom: "1px solid rgba(255,255,255,0.12)",
        padding: "14px 16px",
        textAlign: "left",
        fontSize: "15px",
        color: "#ffffff",
        fontWeight: "700",
        letterSpacing: "0.3px",
    };

    return (
        <section style={pageStyle}>
            <div style={shellStyle}>

                <form onSubmit={handleSubmit} style={formStyle}>
                    <div>
                        <label style={fieldStyle}>Rank</label>
                        <input
                            type="number"
                            value={rank}
                            onChange={(e) => setRank(e.target.value)}
                            style={inputStyle}
                            placeholder="Enter your rank"
                            required
                        />
                    </div>

                    <div>
                        <label style={fieldStyle}>Category</label>
                        <select
                            value={category}
                            onChange={(e) => setCategory(e.target.value)}
                            style={inputStyle}
                            required
                        >
                            <option value="">Select Category</option>
                            <option value="OC">OC</option>
                            <option value="BCA">BCA</option>
                            <option value="BCB">BCB</option>
                            <option value="BCC">BCC</option>
                            <option value="BCD">BCD</option>
                            <option value="BCE">BCE</option>
                            <option value="SC">SC</option>
                            <option value="ST">ST</option>
                            <option value="EWS">EWS</option>
                        </select>
                    </div>

                    <div>
                        <label style={fieldStyle}>Gender</label>
                        <select
                            value={gender}
                            onChange={(e) => setGender(e.target.value)}
                            style={inputStyle}
                        >
                            <option value="FEMALE">Female</option>
                            <option value="MALE">Male</option>
                        </select>
                    </div>

                    <div>
                        <label style={fieldStyle}>Sort By</label>
                        <select
                            value={sortChoice}
                            onChange={(e) => setSortChoice(e.target.value)}
                            style={inputStyle}
                        >
                            <option value="1">Chance</option>
                            <option value="2">Cutoff</option>
                        </select>
                    </div>

                    <MultiSelectDropdown
                        label="Districts"
                        icon="📍"
                        options={districtOptions}
                        selected={districts}
                        setSelected={setDistricts}
                        placeholder="Select one or more districts"
                        accent="#2563eb"
                    />

                    <MultiSelectDropdown
                        label="Branches"
                        icon="🎓"
                        options={branchOptions}
                        selected={branches}
                        setSelected={setBranches}
                        placeholder="Select one or more branches"
                        accent="#7c3aed"
                    />

                    <div>
                        <label style={fieldStyle}>Show only SAFE/TARGET?</label>
                        <select
                            value={showOnlySafeTarget}
                            onChange={(e) => setShowOnlySafeTarget(e.target.value)}
                            style={inputStyle}
                        >
                            <option value="no">No</option>
                            <option value="yes">Yes</option>
                        </select>
                    </div>

                    <div style={{ display: "flex", alignItems: "end" }}>
                        <button type="submit" style={buttonStyle} disabled={loading}>
                            {loading ? "Predicting..." : "Find Colleges"}
                        </button>
                    </div>
                </form>

                {error && <div style={errorStyle}>{error}</div>}

                {results.length > 0 && (
                    <div style={tableWrapStyle}>
                        <table style={tableStyle}>
                            <thead>
                                <tr
                                    style={{
                                        background: "linear-gradient(90deg, #0f4c5c, #164e63)",
                                    }}
                                >
                                    <th style={tableHeaderStyle}>College</th>
                                    <th style={tableHeaderStyle}>Branch</th>
                                    <th style={tableHeaderStyle}>District</th>
                                    <th style={tableHeaderStyle}>Cutoff</th>
                                    <th style={tableHeaderStyle}>Chance</th>
                                    <th style={tableHeaderStyle}>Confidence</th>
                                </tr>
                            </thead>
                            <tbody>
                                {results.map((item, index) => (
                                    <tr
                                        key={index}
                                        style={{
                                            background:
                                                index % 2 === 0 ? "#f8fffc" : "#edf7f3",
                                            transition: "0.2s",
                                        }}
                                        onMouseEnter={(e) => {
                                            e.currentTarget.style.background = "#dff3ec";
                                        }}
                                        onMouseLeave={(e) => {
                                            e.currentTarget.style.background =
                                                index % 2 === 0 ? "#f8fffc" : "#edf7f3";
                                        }}
                                    >
                                        <td style={thTdStyle}>{item.College}</td>
                                        <td style={thTdStyle}>{item.Branch}</td>
                                        <td style={thTdStyle}>{item.District}</td>
                                        <td style={thTdStyle}>{item.Predicted_Cutoff}</td>
                                        <td style={thTdStyle}>{item.Chance}</td>
                                        <td style={thTdStyle}>{item.Confidence}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </section>
    );
}

export default PredictorSteps;