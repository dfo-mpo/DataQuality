import { css, setup, CSSAttribute } from "goober";
import React, { useEffect, useState } from "react";

// import { Tooltip } from "baseui/tooltip";
// import { PLACEMENT } from 'baseui/popover';
import { Button } from "baseui/button";
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
  padding: "0.43rem var(--rtiS)",

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
  // width: "200%",
});

const label = css({
  fontSize: "0.875rem",
  marginBottom: "5px",
  lineHeight: 1.6,
  fontFamily: '"Source Sans", sans-serif',

});

const pair_input = css({
  display: "flex",
  flexDirection: "row",
  alignContent: "center",

  button: {
    color: "var(--text-color)",
    borderRadius: "0.5rem",
    lineHeight: 1.2,
    fontSize: "0.875rem",
    padding: "3px 8px",
    fontWeight: 400,
    margin: "-2px 0 -2px 4px",
    backgroundColor: "var(--btn-color)",
    border: "1px solid var(--btn-boarder-color)",
  },

  "button:hover": {
    backgroundColor: "var(--btn-hover-color)",
  },

  "button:focus": { // overwrite default thick white boarder when clicking
    outline: 0,
  },

  "button:active": { // give clicking effect when user presses button
    backgroundColor: "var(--btn-active-color)"
  },
})

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

  const handleOnClick = (e) => {
    e.stopPropagation();

    // Get inputed text for column pair
    const firstColObj = document.getElementById(name+" col1") as HTMLInputElement;
    const secondColObj = document.getElementById(name+" col2") as HTMLInputElement;
    const firstCol = firstColObj.value.trim();
    const secondCol = secondColObj.value.trim();

    // Verify both columns are entered
    if (firstCol && firstCol !== '' && secondCol && secondCol !== '') {
      // Save pair
      const pair = `"${firstCol}", "${secondCol}"`
      if (tags.length) {
        setTags(tags.slice(0, -1));
      }

      if (tags.includes(pair)) {
        onExisting && onExisting(pair);
        return;
      }
      setTags([...tags, pair]);
      firstColObj.value = "";
      secondColObj.value = "";
      e.preventDefault();

    } 
  };

  const onTagRemove = (text: string) => {
    setTags(tags.filter(tag => tag !== text));
    onRemoved && onRemoved(text);
  };

  return (
    <>
      <p className={cc("rti--label", label)}>{name}</p> 
      <div aria-labelledby={name} className={cc("rti--container", RTIContainer)}>
        {tags.map(tag => (
          <Tag key={tag} text={tag} remove={onTagRemove} />
        ))}
        <div className={cc("rti--pair-input", pair_input)}>
          <Hint options={suggestions} allowTabFill={true}>
            <input
                id={name+" col1"}
                className={cc("rti--input", RTIInput)}
                type="text"
                name={name}
                placeholder={placeHolder}
                onBlur={onBlur}
            />
          </Hint>
          <Hint options={suggestions} allowTabFill={true}>
            <input
                id={name+" col2"}
                className={cc("rti--input", RTIInput)}
                type="text"
                name={name}
                placeholder={placeHolder}
                onBlur={onBlur}
            />
          </Hint>
          <Button
            onClick={handleOnClick}
          >
          Add Pair</Button>
        </div>
      </div>
    </>
  );
};
