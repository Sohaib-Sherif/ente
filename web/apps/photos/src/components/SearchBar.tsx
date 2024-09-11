import { useIsMobileWidth } from "@/base/hooks";
import { FileType } from "@/media/file-type";
import { PeopleList } from "@/new/photos/components/PeopleList";
import {
    isMLEnabled,
    isMLSupported,
    mlStatusSnapshot,
} from "@/new/photos/services/ml";
import type {
    City,
    SearchDateComponents,
    SearchPerson,
    SearchResultSummary,
} from "@/new/photos/services/search/types";
import {
    ClipSearchScores,
    SearchOption,
    SearchQuery,
    Suggestion,
    SuggestionType,
} from "@/new/photos/services/search/types";
import { labelForSuggestionType } from "@/new/photos/services/search/ui";
import type { LocationTag } from "@/new/photos/services/user-entity";
import { EnteFile } from "@/new/photos/types/file";
import {
    FreeFlowText,
    Row,
    SpaceBetweenFlex,
} from "@ente/shared/components/Container";
import CalendarIcon from "@mui/icons-material/CalendarMonth";
import CloseIcon from "@mui/icons-material/Close";
import FolderIcon from "@mui/icons-material/Folder";
import ImageIcon from "@mui/icons-material/Image";
import LocationIcon from "@mui/icons-material/LocationOn";
import SearchIcon from "@mui/icons-material/Search";
import {
    Box,
    Divider,
    IconButton,
    Stack,
    styled,
    Typography,
} from "@mui/material";
import CollectionCard from "components/Collections/CollectionCard";
import { ResultPreviewTile } from "components/Collections/styledComponents";
import { t } from "i18next";
import pDebounce from "p-debounce";
import { AppContext } from "pages/_app";
import {
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useRef,
    useState,
} from "react";
import {
    components as SelectComponents,
    type ControlProps,
    type InputActionMeta,
    type InputProps,
    type MenuProps,
    type OptionProps,
    type SelectInstance,
    type StylesConfig,
} from "react-select";
import AsyncSelect from "react-select/async";
import {
    getAutoCompleteSuggestions,
    getDefaultOptions,
} from "services/searchService";
import { Collection } from "types/collection";

interface SearchBarProps {
    isInSearchMode: boolean;
    setIsInSearchMode: (v: boolean) => void;
    updateSearch: UpdateSearch;
    collections: Collection[];
    files: EnteFile[];
}

export type UpdateSearch = (
    search: SearchQuery,
    summary: SearchResultSummary,
) => void;

export const SearchBar: React.FC<SearchBarProps> = ({
    setIsInSearchMode,
    isInSearchMode,
    ...props
}) => {
    const isMobileWidth = useIsMobileWidth();

    const showSearchInput = () => setIsInSearchMode(true);

    return (
        <Box sx={{ flex: 1, px: ["4px", "24px"] }}>
            {isMobileWidth && !isInSearchMode ? (
                <MobileSearchArea onSearch={showSearchInput} />
            ) : (
                <SearchInput
                    {...props}
                    isOpen={isInSearchMode}
                    setIsOpen={setIsInSearchMode}
                />
            )}
        </Box>
    );
};

interface MobileSearchAreaProps {
    /** Called when the user presses the search button. */
    onSearch: () => void;
}

const MobileSearchArea: React.FC<MobileSearchAreaProps> = ({ onSearch }) => (
    <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
        <IconButton onClick={onSearch}>
            <SearchIcon />
        </IconButton>
    </Box>
);

interface SearchInputProps {
    isOpen: boolean;
    setIsOpen: (value: boolean) => void;
    updateSearch: UpdateSearch;
    files: EnteFile[];
    collections: Collection[];
}

const SearchInput: React.FC<SearchInputProps> = ({
    isOpen,
    setIsOpen,
    updateSearch,
    files,
    collections,
}) => {
    const appContext = useContext(AppContext);

    // A ref to the top level Select.
    const selectRef = useRef(null);
    // The currently selected option.
    const [value, setValue] = useState<SearchOption | undefined>();
    // The contents of the input field associated with the select.
    const [inputValue, setInputValue] = useState("");
    // The default options shown in the select menu when nothing has been typed.
    const [defaultOptions, setDefaultOptions] = useState([]);

    useEffect(() => {
        search(value);
    }, [value]);

    useEffect(() => {
        refreshDefaultOptions();
        const t = setInterval(() => refreshDefaultOptions(), 2000);
        return () => clearInterval(t);
    }, []);

    const handleChange = (value: SearchOption) => {
        setValue(value);
        setInputValue(value?.label);
        // The Select has a blurInputOnSelect prop, but that makes the input
        // field lose focus, not the entire menu (e.g. when pressing twice).
        //
        // We anyways need the ref so that we can blur on selecting a person
        // from the default options.
        selectRef.current?.blur();
    };

    const handleInputChange = (value: string, actionMeta: InputActionMeta) => {
        if (actionMeta.action === "input-change") {
            setInputValue(value);
        }
    };

    const refreshDefaultOptions = async () => {
        setDefaultOptions(await getDefaultOptions());
    };

    const resetSearch = () => {
        if (isOpen) {
            appContext.startLoading();
            updateSearch(null, null);
            setTimeout(() => {
                appContext.finishLoading();
            }, 10);
            setIsOpen(false);
            setValue(null);
            setInputValue("");
        }
    };

    const getOptions = useCallback(
        pDebounce(getAutoCompleteSuggestions(files, collections), 250),
        [files, collections],
    );

    const search = (selectedOption: SearchOption) => {
        if (!selectedOption) {
            return;
        }
        let search: SearchQuery;
        switch (selectedOption.type) {
            case SuggestionType.DATE:
                search = {
                    date: selectedOption.value as SearchDateComponents,
                };
                setIsOpen(true);
                break;
            case SuggestionType.LOCATION:
                search = {
                    location: selectedOption.value as LocationTag,
                };
                setIsOpen(true);
                break;
            case SuggestionType.CITY:
                search = {
                    city: selectedOption.value as City,
                };
                setIsOpen(true);
                break;
            case SuggestionType.COLLECTION:
                search = { collection: selectedOption.value as number };
                setValue(null);
                setInputValue("");
                break;
            case SuggestionType.FILE_NAME:
                search = { files: selectedOption.value as number[] };
                break;
            case SuggestionType.FILE_CAPTION:
                search = { files: selectedOption.value as number[] };
                break;
            case SuggestionType.PERSON:
                search = { person: selectedOption.value as SearchPerson };
                break;
            case SuggestionType.FILE_TYPE:
                search = { fileType: selectedOption.value as FileType };
                break;
            case SuggestionType.CLIP:
                search = { clip: selectedOption.value as ClipSearchScores };
        }
        updateSearch(search, {
            optionName: selectedOption.label,
            fileCount: selectedOption.fileCount,
        });
    };

    // TODO: HACK as AsyncSelect does not support default options reloading on focus/click
    // unwanted side effect: placeholder is not shown on focus/click
    // https://github.com/JedWatson/react-select/issues/1879
    // for correct fix AsyncSelect can be extended to support default options reloading on focus/click
    const handleOnFocus = () => {
        refreshDefaultOptions();
    };

    const handleSelectCGroup = (value: SearchOption) => {
        // Dismiss the search menu.
        selectRef.current?.blur();
        setValue(value);
    };

    const components = useMemo(() => ({ Option, Control, Input }), []);

    return (
        <SearchInputWrapper>
            <AsyncSelect
                ref={selectRef}
                value={value}
                components={components}
                placeholder={t("search_hint")}
                loadOptions={getOptions}
                onChange={handleChange}
                onFocus={handleOnFocus}
                isMulti={false}
                isClearable
                escapeClearsValue
                inputValue={inputValue}
                onInputChange={handleInputChange}
                styles={SelectStyles}
                defaultOptions={isMLEnabled() ? defaultOptions : []}
                noOptionsMessage={({ inputValue }) =>
                    shouldShowEmptyState(inputValue) ? (
                        <EmptyState onSelectCGroup={handleSelectCGroup} />
                    ) : null
                }
            />

            {isOpen && (
                <IconButton onClick={() => resetSearch()} sx={{ ml: 1 }}>
                    <CloseIcon />
                </IconButton>
            )}
        </SearchInputWrapper>
    );
};

const SearchInputWrapper = styled(Box)`
    display: flex;
    width: 100%;
    align-items: center;
    justify-content: center;
    background: ${({ theme }) => theme.colors.background.base};
    max-width: 484px;
    margin: auto;
`;

const SelectStyles: StylesConfig<SearchOption, false> = {
    container: (style) => ({ ...style, flex: 1 }),
    control: (style, { isFocused }) => ({
        ...style,
        // Give a solid background color.
        backgroundColor: "rgb(26, 26, 26)",
        borderColor: isFocused ? "#1DB954" : "transparent",
        boxShadow: "none",
        ":hover": {
            borderColor: "#01DE4D",
            cursor: "text",
        },
    }),
    input: (styles) => ({ ...styles, color: "#fff" }),
    menu: (style) => ({
        ...style,
        // Suppress the default margin at the top.
        marginTop: "1px",
        // Same background color as the control (must be solid, otherwise the
        // content behind the menu shows through).
        backgroundColor: "rgb(26, 26, 26)",
    }),
    option: (style, { isFocused }) => ({
        ...style,
        padding: 0,
        backgroundColor: "transparent !important",
        "& :hover": {
            cursor: "pointer",
        },
        "& .main": {
            backgroundColor: isFocused && "#202020",
        },
        "&:last-child .MuiDivider-root": {
            display: "none",
        },
    }),
    placeholder: (style) => ({
        ...style,
        color: "rgba(255, 255, 255, 0.7)",
        whiteSpace: "nowrap",
    }),
    // Hide some things we don't need.
    dropdownIndicator: (style) => ({ ...style, display: "none" }),
    indicatorSeparator: (style) => ({ ...style, display: "none" }),
    clearIndicator: (style) => ({ ...style, display: "none" }),
};

const Control = ({ children, ...props }: ControlProps<SearchOption, false>) => (
    <SelectComponents.Control {...props}>
        <Stack
            direction="row"
            sx={{
                alignItems: "center",
                // Fill the entire control (the control uses display flex).
                flex: 1,
            }}
        >
            <Box
                sx={{
                    display: "inline-flex",
                    // Match the default padding of the ValueContainer to make
                    // the icon look properly spaced and aligned.
                    pl: "8px",
                    color: (theme) => theme.colors.stroke.muted,
                }}
            >
                {iconForOptionType(props.getValue()[0]?.type)}
            </Box>
            {children}
        </Stack>
    </SelectComponents.Control>
);

const iconForOptionType = (type: SuggestionType | undefined) => {
    switch (type) {
        case SuggestionType.DATE:
            return <CalendarIcon />;
        case SuggestionType.LOCATION:
        case SuggestionType.CITY:
            return <LocationIcon />;
        case SuggestionType.COLLECTION:
            return <FolderIcon />;
        case SuggestionType.FILE_NAME:
            return <ImageIcon />;
        default:
            return <SearchIcon />;
    }
};

/**
 * A preflight check for whether or not we should show the EmptyState.
 *
 * react-select seems to only suppress showing anything at all in the menu if we
 * return `null` from the function passed to `noOptionsMessage`. Returning
 * `false`, or returning `null` from the EmptyState itself doesn't work and
 * causes a empty div to be shown instead.
 */
const shouldShowEmptyState = (inputValue: string) => {
    // Don't show empty state if the user has entered search input.
    if (inputValue) return false;

    // Don't show empty state if there is no ML related information.
    if (!isMLSupported) return false;

    const status = isMLSupported && mlStatusSnapshot();
    if (!status || status.phase == "disabled") return false;

    // Show it otherwise.
    return true;
};

interface EmptyStateProps {
    /** Called when the user selects a cgroup shown in the empty state view. */
    onSelectCGroup: (value: SearchOption) => void;
}

/**
 * The view shown in the menu area when the user has not typed anything in the
 * search box.
 */
const EmptyState: React.FC<EmptyStateProps> = (props) => {
    const status = isMLSupported && mlStatusSnapshot();

    if (!isMLSupported) return null;
    if (!status || status.phase == "disabled") return null;

    let label: string;
    switch (status.phase) {
        case "scheduled":
            label = t("indexing_scheduled");
            break;
        case "indexing":
            label = t("indexing_photos", status);
            break;
        case "fetching":
            label = t("indexing_fetching", status);
            break;
        case "clustering":
            label = t("indexing_people", status);
            break;
        case "done":
            label = t("indexing_done", status);
            break;
    }

    return (
        <Box>
            <Caption>{label}</Caption>
        </Box>
    );

    // const options = props.selectProps.options as SearchOption[];

    // const peopleSuggestions = options.filter(
    //     (o) => o.type === SuggestionType.PERSON,
    // );
    // const people = peopleSuggestions.map((o) => o.value as SearchPerson);

    // const indexStatusSuggestion = options.filter(
    //     (o) => o.type === SuggestionType.INDEX_STATUS,
    // )[0] as Suggestion;

    // const indexStatus = indexStatusSuggestion?.value;
    // return (
    //     <SelectComponents.Menu {...props}>
    //         <Box my={1}>
    //             {isMLEnabled() &&
    //                 indexStatus &&
    //                 (people && people.length > 0 ? (
    //                     <Box>
    //                         <Legend>{t("people")}</Legend>
    //                     </Box>
    //                 ) : (
    //                     <Box height={6} />
    //                 ))}

    //             {isMLEnabled() && indexStatus && (
    //                 <Box>
    //                     <Caption>{indexStatusSuggestion.label}</Caption>
    //                 </Box>
    //             )}
    //             {people && people.length > 0 && (
    //                 <Row>
    //                     <PeopleList
    //                         people={people}
    //                         maxRows={2}
    //                         onSelect={(_, index) => {
    //                             selectRef.current?.blur();
    //                             setSelectedValue(peopleSuggestions[index]);
    //                         }}
    //                     />
    //                 </Row>
    //             )}
    //         </Box>
    //         {props.children}
    //     </SelectComponents.Menu>
    // );

    return <></>;
    return (
        <div>
            <h1>Hello</h1>
            <p>{(console.log(props), props.toString())}</p>
        </div>
    );
};

export async function getMLStatusSuggestion(): Promise<Suggestion> {}

const Option: React.FC<OptionProps<SearchOption, false>> = (props) => (
    <SelectComponents.Option {...props}>
        <LabelWithInfo data={props.data} />
    </SelectComponents.Option>
);

const LabelWithInfo = ({ data }: { data: SearchOption }) => {
    return (
        !data.hide && (
            <>
                <Box className="main" px={2} py={1}>
                    <Typography variant="mini" mb={1}>
                        {labelForSuggestionType(data.type)}
                    </Typography>
                    <SpaceBetweenFlex>
                        <Box mr={1}>
                            <FreeFlowText>
                                <Typography fontWeight={"bold"}>
                                    {data.label}
                                </Typography>
                            </FreeFlowText>
                            <Typography color="text.muted">
                                {t("photos_count", { count: data.fileCount })}
                            </Typography>
                        </Box>

                        <Stack direction={"row"} spacing={1}>
                            {data.previewFiles.map((file) => (
                                <CollectionCard
                                    key={file.id}
                                    coverFile={file}
                                    onClick={() => null}
                                    collectionTile={ResultPreviewTile}
                                />
                            ))}
                        </Stack>
                    </SpaceBetweenFlex>
                </Box>
                <Divider sx={{ mx: 2, my: 1 }} />
            </>
        )
    );
};

type CustomMenuProps = MenuProps<SearchOption, false> & {
    selectRef: React.RefObject<SelectInstance>;
    // Cannot call it setValue since the menu itself already has that.
    setSelectedValue: (value: SearchOption) => void;
};

const CustomMenu: React.FC<CustomMenuProps> = ({
    selectRef,
    setSelectedValue,
    ...props
}) => {
    // Need to cast here, otherwise the react-select types think selectProps can
    // also be something that supports multiple selection groups.
    const options = props.selectProps.options as SearchOption[];

    const peopleSuggestions = options.filter(
        (o) => o.type === SuggestionType.PERSON,
    );
    const people = peopleSuggestions.map((o) => o.value as SearchPerson);

    const indexStatusSuggestion = options.filter(
        (o) => o.type === SuggestionType.INDEX_STATUS,
    )[0] as Suggestion;

    const indexStatus = indexStatusSuggestion?.value;
    return (
        <SelectComponents.Menu {...props}>
            <Box my={1}>
                {isMLEnabled() &&
                    indexStatus &&
                    (people && people.length > 0 ? (
                        <Box>
                            <Legend>{t("people")}</Legend>
                        </Box>
                    ) : (
                        <Box height={6} />
                    ))}

                {isMLEnabled() && indexStatus && (
                    <Box>
                        <Caption>{indexStatusSuggestion.label}</Caption>
                    </Box>
                )}
                {people && people.length > 0 && (
                    <Row>
                        <PeopleList
                            people={people}
                            maxRows={2}
                            onSelect={(_, index) => {
                                selectRef.current?.blur();
                                setSelectedValue(peopleSuggestions[index]);
                            }}
                        />
                    </Row>
                )}
            </Box>
            {props.children}
        </SelectComponents.Menu>
    );
};

const Legend = styled("span")`
    font-size: 20px;
    color: #ddd;
    display: inline;
    padding: 0px 12px;
`;

const Caption = styled("span")`
    font-size: 12px;
    display: inline;
    padding: 0px 12px;
`;

// A custom input for react-select that is always visible. This is a roundabout
// hack the existing code used to display the search string when showing the
// results page; likely there should be a better way.
const Input: React.FC<InputProps<SearchOption, false>> = (props) => (
    <SelectComponents.Input {...props} isHidden={false} />
);
