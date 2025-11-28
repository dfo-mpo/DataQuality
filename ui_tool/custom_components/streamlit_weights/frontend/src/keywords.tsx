import { ResizeObserver } from '@juggle/resize-observer';
import React,{ useEffect, useRef, useState }  from "react"
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
  let { label, initialValue, step, min, max}: PythonArgs = props.args;
  const [value, setValue] = useState(initialValue);
  const rootRef = useRef<HTMLDivElement>(null);

  // Sync local state when Python sends a new initialValue (e.g. metrics changed)   .hasOwnProperty
  useEffect(() => {
    const newValue = initialValue;

    for (const key in newValue) {
      if (value.hasOwnProperty(key)) {
        newValue[key] = value[key]
      }
    }
    setValue(newValue);  
    Streamlit.setFrameHeight();  
  }, [initialValue]);  

  const onSubmit = (values: Record<string, number>) => {
    setValue(values)
    Streamlit.setComponentValue((values))
  }

  // Observe height changes of the wrapper div  
  useEffect(() => {  
    if (!rootRef.current || typeof ResizeObserver === "undefined") return;  
  
    const ro = new ResizeObserver(() => {  
      Streamlit.setFrameHeight(); // lets the lib compute document height  
    });  
  
    ro.observe(rootRef.current);  
    return () => ro.disconnect();  
  }, []);  

  // Keep height in sync on mount  
  useEffect(() => {  
    Streamlit.setFrameHeight();  
  }, []); 

  return (
    <div ref={rootRef}>
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
