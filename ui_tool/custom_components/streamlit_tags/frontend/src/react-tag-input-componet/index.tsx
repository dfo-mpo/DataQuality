import { css, setup, CSSAttribute } from "goober";
import React, { useEffect, useState } from "react";

// import { Tooltip } from "baseui/tooltip";
// import { PLACEMENT } from 'baseui/popover';
import cc from "./classnames";
import Tag from "./tag";
import {Hint} from "../react-autocomplete-hint";



export interface IHintOption {
    id: string | number;
    label: string;
}

export interface TagsInputProps {
  name?: string;
  placeHolder?: string;
  value: string[];
  onChange?: (tags: string[]) => void;
  suggestions: Array<string> | Array<IHintOption>;
  onBlur?: any;
  separators?: string[];
  onExisting?: (tag: string) => void;
  onRemoved?: (tag: string) => void;
  maxTags: number;
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
  padding: "var(--rtiS)",

  "&:focus-within": {
    // borderColor: "var(--rtiMain)",
    border: "1px solid #FF3333",
  },
});

const RTIInput = css({
  border: 0,
  outline: 0,
  fontSize: "inherit",
  lineHeight: "inherit",
  width: "100%",
});

const label = css({
  fontSize: "0.875rem",
  marginBottom: "5px",
  lineHeight: 1.6,
  fontFamily: '"Source Sans 3", sans-serif',

});

const hintIcon = css({
  stroke: "var(--help-icon-color)",
  strokeWidth: 2.25,
} as unknown as CSSAttribute);

const defaultSeprators = ["Enter"];

export const TagsInput = ({
  name,
  placeHolder,
  value,
  onChange,
  onBlur,
  separators,
  onExisting,
  onRemoved,
  suggestions,
  maxTags
}: TagsInputProps) => {
  let [tags, setTags] = useState(value || []);

  useEffect(() => {
    onChange && onChange(tags);
  }, [tags]);

  if (maxTags >= 0) {
      let remainingLimit = Math.max(maxTags, 0)
      tags = tags.slice(0, remainingLimit)
  }

  const handleOnKeyUp = (e) => {
    e.stopPropagation();

    const text = e.target.value;

    if (e.key === "Backspace" && tags.length && !text) {
      setTags(tags.slice(0, -1));
    }


    if (text && (separators || defaultSeprators).includes(e.key)) {
      if (tags.includes(text)) {
        onExisting && onExisting(text);
        return;
      }
      setTags([...tags, text]);
      e.target.value = "";
      e.preventDefault();
    }
  };

  const onTagRemove = (text: string) => {
    setTags(tags.filter(tag => tag !== text));
    onRemoved && onRemoved(text);
  };

  return (
    <>
      {/* <div style={{display: "flex", flexDirection: "row", justifyContent: "space-between"}}> */}
      <p className={cc("rti--label", label)}>{name}</p>
        {/* <Tooltip 
        isOpen={true}
        content={"test test test tes t sdfas fha askdj fha fhasij nfase f asdfjkl dfjei f sdklfjasioefjasie fk dfl eif k"}  
        placement={PLACEMENT.topLeft}
        overrides={{  
          Body: { style: { zIndex: 1000, right: 0 } },  
          Inner: { style: { maxWidth: "350px" } },  
        }} >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" 
          fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" className={cc("hint--icon", hintIcon)}>
            <circle cx="12" cy="12" r="10"></circle>
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
            <line x1="12" y1="17" x2="12.01" y2="17"></line>
          </svg>
        </Tooltip> */}
        
      {/* </div> */}
      <div aria-labelledby={name} className={cc("rti--container", RTIContainer)}>
        {tags.map(tag => (
          <Tag key={tag} text={tag} remove={onTagRemove} />
        ))}

        <Hint options={suggestions} allowTabFill={true}>
            <input
                className={cc("rti--input", RTIInput)}
                type="text"
                name={name}
                placeholder={placeHolder}
                onKeyDown={handleOnKeyUp}
                onBlur={onBlur}
        />
      </Hint>
      </div>
    </>
    
  );
};
