import {
    LocalizationProvider,
    MobileDateTimePicker,
} from "@mui/x-date-pickers";
import { AdapterDayjs } from "@mui/x-date-pickers/AdapterDayjs";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import React, { useState } from "react";

interface PhotoDateTimePickerProps {
    /**
     * The initial date to preselect in the date/time picker.
     *
     * If not provided, the current date/time is used.
     */
    initialValue?: Date;
    /**
     * If true, then the picker shows provided date/time but doesn't allow
     * editing it.
     */
    disabled?: boolean;
    /**
     * Callback invoked when the user makes and confirms a date/time.
     */
    onAccept: (date: Date) => void;
    /**
     * Optional callback invoked when the picker has been closed.
     */
    onClose?: () => void;
}

// [Note: Obtaining UTC dates from MUI's x-date-picker]
//
// See: https://mui.com/x/react-date-pickers/timezone/
dayjs.extend(utc);

/**
 * A customized version of MUI DateTimePicker suitable for use in selecting and
 * modifying the date/time for a photo.
 */
export const PhotoDateTimePicker: React.FC<PhotoDateTimePickerProps> = ({
    initialValue,
    disabled,
    onAccept,
    onClose,
}) => {
    const [open, setOpen] = useState(true);
    const [value, setValue] = useState<Date | null>(
        initialValue ?? dayjs.utc(),
    );

    const handleAccept = (date: Date | null) => {
        console.log({ date });
        date && onAccept(date);
    };

    const handleClose = () => {
        setOpen(false);
        onClose?.();
    };

    return (
        <LocalizationProvider dateAdapter={AdapterDayjs}>
            <MobileDateTimePicker
                value={value}
                onChange={setValue}
                open={open}
                onClose={handleClose}
                onOpen={() => setOpen(true)}
                disabled={disabled}
                minDateTime={new Date(1800, 0, 1)}
                disableFuture={true}
                timezone={"UTC"}
                onAccept={handleAccept}
                DialogProps={{
                    sx: {
                        zIndex: "1502",
                        ".MuiPickersToolbar-penIconButton": {
                            display: "none",
                        },
                        ".MuiDialog-paper": { width: "320px" },
                        ".MuiClockPicker-root": {
                            position: "relative",
                            minHeight: "292px",
                        },
                        ".PrivatePickersSlideTransition-root": {
                            minHeight: "200px",
                        },
                    },
                }}
                renderInput={() => <></>}
            />
        </LocalizationProvider>
    );
};
