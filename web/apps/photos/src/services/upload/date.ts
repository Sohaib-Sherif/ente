import log from "@/base/log";

/**
 * Try to extract a date (as epoch microseconds) from a file name by matching it
 * against certain known patterns for media files.
 *
 * This uses all sorts of arbitrary heuristics gathered over time from feedback
 * by users. In particular, this is meant to capture the dates from screenshots
 * and chat app forwards.
 *
 * If the filename doesn't match a known pattern, or if there is some error
 * during the parsing, return `undefined`.
 */
export const tryParseEpochMicrosecondsFromFileName = (
    fileName: string,
): number | undefined => {
    try {
        fileName = fileName.trim();
        let parsedDate: Date;
        if (fileName.startsWith("IMG-") || fileName.startsWith("VID-")) {
            // WhatsApp media files
            // Sample name: IMG-20171218-WA0028.jpg
            parsedDate = parseDateFromFusedDateString(fileName.split("-")[1]);
        } else if (fileName.startsWith("Screenshot_")) {
            // Screenshots on Android
            // Sample name: Screenshot_20181227-152914.jpg
            parsedDate = parseDateFromFusedDateString(
                fileName.replaceAll("Screenshot_", ""),
            );
        } else if (fileName.startsWith("signal-")) {
            // Signal images
            // Sample name: signal-2018-08-21-100217.jpg
            const p = fileName.split("-");
            const dateString = `${p[1]}${p[2]}${p[3]}-${p[4]}`;
            parsedDate = parseDateFromFusedDateString(dateString);
        }
        if (!parsedDate) {
            parsedDate = parseDateFromDigitGroups(fileName);
        }
        return validateAndGetCreationUnixTimeInMicroSeconds(parsedDate);
    } catch (e) {
        log.error(`Could not extract date from file name ${fileName}`, e);
        return undefined;
    }
};

export function validateAndGetCreationUnixTimeInMicroSeconds(dateTime: Date) {
    if (!dateTime || isNaN(dateTime.getTime())) {
        return null;
    }
    const unixTime = dateTime.getTime() * 1000;
    //ignoring dateTimeString = "0000:00:00 00:00:00"
    if (unixTime === Date.UTC(0, 0, 0, 0, 0, 0, 0) || unixTime === 0) {
        return null;
    } else if (unixTime > Date.now() * 1000) {
        return null;
    } else {
        return unixTime;
    }
}

const currentYear = new Date().getFullYear();

/**
 * An intermediate data structure we use for the functions in this file. It
 * stores the various components of a JavaScript date.
 *
 * In particular, the month is 0-indexed, as it is for the JavaScript Date's
 * `getMonth`.
 */
interface DateComponents {
    year: number;
    month: number;
    day: number;
    hour: number;
    minute: number;
    second: number;
}

/**
 * Parse a date from a string of the form "YYYYMMDD-HHMMSS".
 */
const parseDateFromFusedDateString = (s: string) =>
    validateAndGetDateFromComponents({
        year: Number(s.slice(0, 4)),
        // JavaScript Date's month is 0-indexed.
        month: Number(s.slice(4, 6)) - 1,
        day: Number(s.slice(6, 8)),
        hour: Number(s.slice(9, 11)),
        minute: Number(s.slice(11, 13)),
        second: Number(s.slice(13, 15)),
    });

/**
 * Try to see if we can parse an date from a string with arbitrary separators.
 *
 * For example, consider a string like "2018-08-19 12:34:45". We see if it is
 * possible to extract six symbol separated digit groups from the string. If so,
 * we use them to form a date.
 */
export const parseDateFromDigitGroups = (s: string) => {
    const [year, month, day, hour, minute, second] = s.match(/\d+/g) ?? [];

    // If the filename has 8 consecutive and then 6 consecutive digits, then
    // there is a high possibility that it is a date of the form
    // "YYYYMMDD-HHMMSS".
    if (year?.length == 8 && month?.length == 6) {
        return parseDateFromFusedDateString(year + "-" + month);
    }

    return validateAndGetDateFromComponents({
        year: Number(year),
        // JavaScript Date's month is 0-indexed.
        month: Number(month) - 1,
        day: Number(day),
        hour: Number(hour),
        minute: Number(minute),
        second: Number(second),
    });
};

function validateAndGetDateFromComponents(
    dateComponents: DateComponents,
    options = { minYear: 1990, maxYear: currentYear + 1 },
) {
    let date = getDateFromComponents(dateComponents);
    if (
        hasTimeValues(dateComponents) &&
        !isTimePartValid(date, dateComponents)
    ) {
        // if the date has time values but they are not valid
        // then we remove the time values and try to validate the date
        date = getDateFromComponents(removeTimeValues(dateComponents));
    }
    if (!isDatePartValid(date, dateComponents)) {
        return null;
    }
    if (
        date.getFullYear() < options.minYear ||
        date.getFullYear() > options.maxYear
    ) {
        return null;
    }
    return date;
}

function isTimePartValid(date: Date, dateComponents: DateComponents) {
    return (
        date.getHours() === dateComponents.hour &&
        date.getMinutes() === dateComponents.minute &&
        date.getSeconds() === dateComponents.second
    );
}

function isDatePartValid(date: Date, dateComponents: DateComponents) {
    return (
        date.getFullYear() === dateComponents.year &&
        date.getMonth() === dateComponents.month &&
        date.getDate() === dateComponents.day
    );
}

function getDateFromComponents(dateComponents: DateComponents) {
    const { year, month, day, hour, minute, second } = dateComponents;
    if (hasTimeValues(dateComponents)) {
        return new Date(year, month, day, hour, minute, second);
    } else {
        return new Date(year, month, day);
    }
}

function hasTimeValues(dateComponents: DateComponents) {
    const { hour, minute, second } = dateComponents;
    return !isNaN(hour) && !isNaN(minute) && !isNaN(second);
}

function removeTimeValues(dateComponents: DateComponents): DateComponents {
    return { ...dateComponents, hour: 0, minute: 0, second: 0 };
}
