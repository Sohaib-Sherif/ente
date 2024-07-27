import type { Metadata } from "./types/file";

export const hasFileHash = (file: Metadata) =>
    !!file.hash || (!!file.imageHash && !!file.videoHash);

/**
 * [Note: Photos are always in local date/time]
 *
 * Photos out in the wild frequently do not have associated timezone offsets for
 * the date/time embedded in their metadata. This is a artifact of an era where
 * cameras didn't know often even know their date/time correctly, let alone the
 * UTC offset of their local data/time.
 *
 * This is beginning to change with smartphone cameras, and is even reflected in
 * the standards. e.g. Exif metadata now has auxillary "OffsetTime*" tags for
 * indicating the UTC offset of the local date/time in the existing Exif tags
 * (See: [Note: Exif dates]).
 *
 * So a photos app needs to deal with a mixture of photos whose dates may or may
 * not have UTC offsets. This is fact #1.
 *
 * Users expect to see the time they took the photo, not the time in the place
 * they are currently. People expect a New Year's Eve photo from a vacation to
 * show up as midnight, not as (e.g.) 19:30 IST. This is fact #2.
 *
 * Combine these two facts, and if you ponder a bit, you'll find that there is
 * only one way for a photos app to show / sort / label the date – by using the
 * local date/time without the attached UTC offset, **even if it is present**.
 *
 * The UTC offset is still useful though, and we don't want to lose that
 * information. The number of photos with a UTC offset will only increase. And
 * whenever it is present, it provides additional context for the user.
 *
 * So we keep both the local date/time string (an ISO 8601 string guaranteed to
 * be without an associated UTC offset), and an (optional) UTC offset string.
 *
 * It is important to NOT think of the local date/time string as an instant of
 * time that can be converted to a UTC timestamp. It cannot be converted to a
 * UTC timestamp because while it itself might have the optional associated UTC
 * offset, its siblings photos might not. The only way to retain their
 * comparability is to treat them all the "time zone where the photo was taken".
 *
 * Finally, while this is all great, we still have existing code that deals with
 * UTC timestamps. So we also retain the existing `creationTime` UTC timestamp,
 * but this should be considered deprecated, and over time we should move
 * towards using the `dateTime` string.
 */
export interface ParsedMetadataDate {
    /**
     * A local date/time.
     *
     * This is a partial ISO 8601 date/time string guaranteed not to have a
     * timezone offset. e.g. "2023-08-23T18:03:00.000"
     */
    dateTime: string;
    /**
     * An optional offset from UTC.
     *
     * This is an optional UTC offset string of the form "±HH:mm" or "Z",
     * specifying the timezone offset for {@link dateTime} when available.
     */
    offsetTime: string | undefined;
    /**
     * UTC epoch microseconds derived from {@link dateTime} and
     * {@link offsetTime}.
     *
     * When the {@link offsetTime} is present, this will accurately reflect a
     * UTC timestamp. When the {@link offsetTime} is not present it convert to a
     * UTC timestamp by assuming that the given {@link dateTime} is in the local
     * time where this code is running. This is a good assumption but not always
     * correct (e.g. vacation photos).
     */
    timestamp: number;
}

/**
 * Parse a partial or full ISO 8601 string into a {@link ParsedMetadataDate}.
 *
 * @param s A partial or full ISO 8601 string. That is, it is a string of the
 * form "2023-08-23T18:03:00.000+05:30" or "2023-08-23T12:33:00.000Z" with all
 * components except the year potentially missing.
 *
 * @return A {@link ParsedMetadataDate}, or `undefined` if {@link s} cannot be
 * parsed.
 *
 * ---
 * Some examples:
 *
 * -   "2023"                          => ("2023-01-01T00:00:00.000", undefined)
 * -   "2023-08"                       => ("2023-08-01T00:00:00.000", undefined)
 * -   "2023-08-23"                    => ("2023-08-23T00:00:00.000", undefined)
 * -   "2023-08-23T18:03:00"           => ("2023-08-23T18:03:00.000", undefined)
 * -   "2023-08-23T18:03:00+05:30"     => ("2023-08-23T18:03:00.000', "+05:30")
 * -   "2023-08-23T18:03:00.000+05:30" => ("2023-08-23T18:03:00.000", "+05:30")
 * -   "2023-08-23T12:33:00.000Z"      => ("2023-08-23T12:33:00.000", "Z")
 */
export const parseMetadataDate = (
    s: string,
): ParsedMetadataDate | undefined => {
    // Construct the timestamp using the original string itself. If s is
    // parseable as a date, then this'll be give us the correct UTC timestamp.
    // If the UTC offset is not present, then this will be in the local
    // (current) time.
    const timestamp = new Date(s).getTime() * 1000;
    if (isNaN(timestamp)) {
        // s in not a well formed ISO 8601 date time string.
        return undefined;
    }

    // Now we try to massage s into two parts - the local date/time string, and
    // an UTC offset string.

    let offsetTime: string | undefined;
    let sWithoutOffset: string;

    // Check to see if there is a time-zone descriptor of the form "Z" or
    // "±05:30" or "±0530" at the end of s.
    const m = s.match(/Z|[+-]\d\d:?\d\d$/);
    if (m?.index) {
        sWithoutOffset = s.substring(0, m.index);
        offsetTime = s.substring(m.index);
    } else {
        sWithoutOffset = s;
    }

    // Convert sWithoutOffset - a potentially partial ISO 8601 string - to a
    // canonical ISO 8601 string.
    //
    // In its full generality, this is non-trivial. The approach we take is:
    //
    // 1.   Rely on the browser to be able to partial ISO 8601 string. This
    //      relies on non-standard behaviour but works in practice seemingly.
    //
    // 2.   Get an ISO 8601 representation of it. This is standard.
    //
    // A thing to watch out for is that browsers treat date only and date time
    // strings differently when the offset is not present (as would be for us).
    //
    // > When the time zone offset is absent, date-only forms are interpreted as
    // > a UTC time and date-time forms are interpreted as local time.
    // >
    // > https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date#date_time_string_format
    //
    // For our purpose, we want to always interpret them as UTC time. This way,
    // when we later gets back its string representation for step 2, we will get
    // back the same numerical value, and can just chop off the "Z".
    //
    // So if the length of the string is less than or equal to yyyy-mm-dd (10),
    // then we use it verbatim, otherwise we append a "Z".

    const date = new Date(
        sWithoutOffset + (sWithoutOffset.length <= 10 ? "" : "Z"),
    );

    // The string returned by `toISOString` is guaranteed to be UTC and denoted
    // by the suffix "Z". If we chop that off, we get back a canonical
    // representation we wish for: A otherwise well-formed ISO 9601 string but
    // any time zone descriptor.
    const dateTime = dropLast(date.toISOString());

    return { dateTime, offsetTime, timestamp };
};

const dropLast = (s: string) => (s ? s.substring(0, s.length - 1) : s);
