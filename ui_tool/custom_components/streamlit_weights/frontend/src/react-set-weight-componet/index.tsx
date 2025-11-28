import { css, setup, CSSAttribute } from "goober";
import React, { useEffect, useState } from "react";

import cc from "./classnames";
import Weight from "./weight";

export interface WeightsInputProps {
  name?: string;
  value: Record<string, number>;
  step: number;
  onChange?: (weights: Record<string, number>) => void;
  min?: number;
  max?: number;
  onBlur?: any;
  onExisting?: (tag: string) => void;
}

// initialize goober once
setup(React.createElement);

const RTIContainer = css({
  "--rtiBg": "#fff",
  "--rtiBorder": "transparent",
  "--rtiMain": "var(--primary-color)",
  "--rtiRadius": "0.375rem",
  "--rtiS": "0.5rem",
  "--rtiTag": "#edf2f7",
  "--rtiTagRemove": "#e53e3e",

  "*": {
    boxSizing: "border-box",
    transition: "all 0.2s ease",
  },

  alignItems: "center",
  bg: "var(--rtiBg)",
  border: "1px solid var(--rtiBorder)",
  borderRadius: "var(--rtiRadius)",
  display: "flex",
  flexWrap: "wrap",
  gap: "var(--rtiS)",
  lineHeight: 1.4,
  padding: "0.365rem var(--rtiS)",

  "&:focus-within": {
    // borderColor: "var(--rtiMain)",
    border: "1px solid #FF3333",
  },
});

const label = css({
  fontSize: "0.875rem",
  marginBottom: "6px",
  lineHeight: 1.6,
  fontFamily: '"Source Sans", sans-serif',
});

const totalDisplay = css({
  display: "flex",
  flexDirection: "row",
  width: '100%',
  paddingTop: "5px",

  "& p": {
    paddingLeft: '7%',
    paddingRight: '2%'
  }
});

const redText = css({
  color: 'rgb(255, 75, 75)',
});

export const WeightsInput = ({
  name,
  value,
  step,
  onChange,
  min,
  max,
  onBlur,
  onExisting,
}: WeightsInputProps) => {
  let [weights, setWeights] = useState(value ?? {});
  const [totalWeight, setTotalWeight] = useState(0)
  const decimals = decimalPlaces(step);

  // Helper function to get the number of decimal places of a number
  function decimalPlaces(n: number): number {  
    if (!Number.isFinite(n)) return 0;  
    const s = Math.abs(n).toString(); // e.g., "12.345"  
    const dot = s.indexOf(".");  
    const d = dot >= 0 ? s.length - dot - 1 : 0;  
    // Use a conservative cap for toFixed compatibility  
    return Math.max(0, Math.min(20, d));  
  }

  const handleInputChange = (key: string, next: number) => {
    console.log(next);
    let newWeights = weights;
    newWeights[key] = next;
    setWeights(newWeights);
    setTotalWeight(Number(Object.values(weights).reduce((sum, v) => sum + v, 0).toFixed(decimals)));
    onChange && onChange(weights);
  }

  // Sync when parent prop changes (e.g., metrics added/removed)
  useEffect(() => {
    setWeights(value ?? {});
  }, [value]);

  useEffect(() => {
    setTotalWeight(Number(Object.values(weights).reduce((sum, v) => sum + v, 0).toFixed(decimals)));  
    onChange && onChange(weights);
  }, [weights]);

  return (
    <>
      <p className={cc("rti--label", label)}>{name}</p> 
      <div aria-labelledby={name} className={cc("rti--container", RTIContainer)}>
        {Object.entries(weights).map(([key, val]) => (
          <Weight text={key} value={val} step={step} onInputChange={handleInputChange} min={min} max={max} decimals={decimals}/>
        ))}
        <div className={totalDisplay}>
          <p>Total</p>
          <p className={totalWeight === 1.0 ? '' : redText}>
            {totalWeight === 1.0 ? totalWeight : `${totalWeight}; Weights do not add up to 1.0, using default weights instead!`}
          </p>
        </div>
      </div>
    </>
  );
};
