import React,{ useEffect, useState }  from "react"
import { ComponentProps, Streamlit, withStreamlitConnection } from "streamlit-component-lib"
import { WeightsInput } from "./react-set-weight-componet";
import "./styles.css";

interface PythonArgs {
  label: string
  initialValue: Record<string, number>
  step: number
  min?: number
  max?: number
}

const CustomKeywords = (props: ComponentProps) => {
  // Destructure using Typescript interface
  // This ensures typing validation for received props from Python
  let { label, initialValue, step, min, max}: PythonArgs = props.args
  const [value, setValue] = useState(initialValue)

  const onSubmit = (values: {}) => {
    setValue(values)
    Streamlit.setComponentValue((values))
  }
  useEffect(() => Streamlit.setFrameHeight())
  return (
    <div>
        <WeightsInput
          value={value}
          onChange= {(value) => onSubmit(value)}
          name={label}
          step={step}
          min={min}
          max={max}
      />
    </div>
  )
}

export default withStreamlitConnection(CustomKeywords)
