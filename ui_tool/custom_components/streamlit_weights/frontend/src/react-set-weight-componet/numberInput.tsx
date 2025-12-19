import React,{ useState }  from "react"
import { css } from "goober";
import cc from "./classnames";
import { Button } from "baseui/button";

interface InputProps {
  initialValue: number;
  step: number;
  decimals: number;
  onChange: (next: number) => void;
  min?: number;
  max?: number;
}

const inputContainerStyles = css({
  alignItems: "center",
  borderRadius: "var(--rtiRadius)",
  display: "inline-flex",
  justifyContent: "center",
//   marginLeft: "var(--rtiS)",
  paddingLeft: "var(--rtiS)",
  width: "70%",
  minWidth: '155px',
  overflow: "hidden",
  backgroundColor: "var(--weight-input-background-color)",

  button: {
    background: "none",
    border: 0,
    cursor: "pointer",
    lineHeight: "inherit",
    height: '100%',
    padding: "0 var(--rtiS)",
    fontSize: "0.625em",
    fontWeight: "600",

    "&:hover": {
      background: "var(--rtiTag)",
    },

    "&:active": {
      background: "none",
    },

    "&:focus": {
      outline: "none",
    },

    "&:focus-visible": {
      outline: "none",
    },
  },
});

const inputStyles = css({
    width: "100%",
    minWidth: "100px",
    border: "none",
    background: "none",

    /* Hide arrow keys */
    appearance: "textfield",
    MozAppearance: "textfield",
    "&::-webkit-outer-spin-button": {
        WebkitAppearance: "none",
        margin: 0,
    },
    "&::-webkit-inner-spin-button": {
        WebkitAppearance: "none",
        margin: 0,
    },

    "&:focus-visible": { // overwrite default thick white boarder when clicking
        outline: 0
    },   
});

const svgStyles = css({
    width: '8px',
    height: '8px',
    color: 'var(--button-color)',

    "&:focus": {
        outline: 'none'
    }
});

export default function Input({ initialValue, step, decimals, onChange, min, max }: InputProps) {
    const [value, setValue] = useState(initialValue);

    // Clamp a weight between specified min and max. If not provided then all numbers are allowed
    const clamp = (n: number) => Math.max(min ?? -Infinity, Math.min(max ?? Infinity, n));

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const str = e.target.value.trim(); // it's a string
        // if (str === "") return; // ignore empty clears in a controlled input
        const n = Number(str);
        if (!Number.isNaN(n)) {
            const clamped = clamp(n);
            setValue(clamped);
            onChange(clamp(clamped));
        }
    };

    // Apply change in step
    const onStep = (newValue: number) => {
        let clamped = clamp(newValue);

        // Clip number to prevent floating point error
        clamped = Number(clamped.toFixed(decimals));

        setValue(clamped);
        onChange(clamp(clamped));
    };

    // Generate number input feild with '-' and '+' buttons
    return (
        <div className={inputContainerStyles}>
            <input type="number" className={inputStyles} value={value} onChange={handleInputChange}/>
            <Button onClick={() => onStep(value - step)}>
                <svg viewBox="0 0 8 8" aria-hidden="true" focusable="false" fill="currentColor" xmlns="http://www.w3.org/2000/svg" color="inherit" className={svgStyles}>
                    <path d="M0 3v2h8V3H0z"></path>
                </svg>
            </Button>
            <Button onClick={() => onStep(value + step)}>
                <svg viewBox="0 0 8 8" aria-hidden="true" focusable="false" fill="currentColor" xmlns="http://www.w3.org/2000/svg" color="inherit" className={svgStyles}>
                    <path d="M3 0v3H0v2h3v3h2V5h3V3H5V0H3z"></path>
                </svg>
            </Button>
        </div>
    );
}