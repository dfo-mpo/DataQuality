import React from "react";
import { css } from "goober";
import cc from "./classnames";
import Input from "./numberInput";

interface WeightProps {
  text: string;
  value: number;
  step: number;
  decimals: number;
  onInputChange: (key: string, next: number) => void;
  min?: number;
  max?: number;
}

const tagStyles = css({
  alignItems: "center",
  background: "var(--rtiTag)",
  borderRadius: "var(--rtiRadius)",
  display: "inline-flex",
  justifyContent: "center",
  padding: "4px 10px",
});

const weightContainerStyles = css({
  display: "flex",
  flexDirection: "row",
  justifyContent: "space-evenly",
  width: "100%"
})

export default function Weight({ text, value, step, decimals, onInputChange, min, max }: WeightProps) {
  // Add metric name to onChange function
  const onChange = (next: number) => {
    console.log(next);
    onInputChange(text, next);
  };

  return (
    <div className={weightContainerStyles}>
      <span className={cc("rti--tag", tagStyles)}>{text}</span>
      <Input initialValue={value} step={step} onChange={onChange} min={min} max={max} decimals={decimals}/>
    </div>
  );
}
