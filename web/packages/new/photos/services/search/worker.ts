// TODO-cgroups
/* eslint-disable @typescript-eslint/no-non-null-assertion */
/* eslint-disable @typescript-eslint/prefer-includes */
/* eslint-disable @typescript-eslint/no-unnecessary-condition */
import { getUICreationDate } from "@/media/file-metadata";
import type { EnteFile } from "@/new/photos/types/file";
import { wait } from "@/utils/promise";
import { nullToUndefined } from "@/utils/transform";
import { getPublicMagicMetadataSync } from "@ente/shared/file-metadata";
import type { Component } from "chrono-node";
import * as chrono from "chrono-node";
import { expose } from "comlink";
import type {
    City,
    DateSearchResult,
    Location,
    LocationTagData,
    SearchDateComponents,
    SearchQuery,
    Suggestion,
} from "./types";
import { SuggestionType } from "./types";

/**
 * A web worker that runs the search asynchronously so that the main thread
 * remains responsive.
 */
export class SearchWorker {
    private enteFiles: EnteFile[] = [];

    /**
     * Set the files that we should search across.
     */
    setEnteFiles(enteFiles: EnteFile[]) {
        this.enteFiles = enteFiles;
    }

    /**
     * Convert a search string into a reusable query.
     */
    async createSearchQuery(
        searchString: string,
        locale: string,
        holidays: DateSearchResult[],
    ) {
        return createSearchQuery(searchString, locale, holidays);
    }

    /**
     * Return {@link EnteFile}s that satisfy the given {@link searchQuery}.
     */
    search(searchQuery: SearchQuery) {
        return this.enteFiles.filter((f) => isMatch(f, searchQuery));
    }
}

expose(SearchWorker);

const createSearchQuery = async (
    searchString: string,
    locale: string,
    holidays: DateSearchResult[],
): Promise<Suggestion[]> => {
    // Normalize it by trimming whitespace and converting to lowercase.
    const s = searchString.trim().toLowerCase();
    if (s.length == 0) return [];

    // TODO Temp
    await wait(0);
    return [dateSuggestion(s, locale, holidays)].flat();
};

const dateSuggestion = (
    s: string,
    locale: string,
    holidays: DateSearchResult[],
) =>
    parseDateComponents(s, locale, holidays).map(({ components, label }) => ({
        type: SuggestionType.DATE,
        value: components,
        label,
    }));

/**
 * Try to parse an arbitrary search string into sets of date components.
 *
 * e.g. "December 2022" will be parsed into a
 *
 *     [(year 2022, month 12, day undefined)]
 *
 * while "22 December 2022" will be parsed into
 *
 *     [(year 2022, month 12, day 22)]
 *
 * In addition, also return a formatted representation of the "best" guess at
 * the date that was intended by the search string.
 */
const parseDateComponents = (
    s: string,
    locale: string,
    holidays: DateSearchResult[],
): DateSearchResult[] =>
    [
        parseChrono(s, locale),
        parseYearComponents(s),
        parseHolidayComponents(s, holidays),
    ].flat();

const parseChrono = (s: string, locale: string): DateSearchResult[] =>
    chrono
        .parse(s)
        .map((result) => {
            const p = result.start;
            const component = (s: Component) =>
                p.isCertain(s) ? nullToUndefined(p.get(s)) : undefined;

            const year = component("year");
            const month = component("month");
            const day = component("day");
            const weekday = component("weekday");
            const hour = component("hour");

            if (!year && !month && !day && !weekday && !hour) return undefined;
            const components = { year, month, day, weekday, hour };

            const format: Intl.DateTimeFormatOptions = {};
            if (year) format.year = "numeric";
            if (month) format.month = "long";
            if (day) format.day = "numeric";
            if (weekday) format.weekday = "long";
            if (hour) {
                format.hour = "numeric";
                format.dayPeriod = "short";
            }

            const formatter = new Intl.DateTimeFormat(locale, format);
            const label = formatter.format(p.date());
            return { components, label };
        })
        .filter((x) => x !== undefined);

/** chrono does not parse years like "2024", so do it manually. */
const parseYearComponents = (s: string): DateSearchResult[] => {
    // s is already trimmed.
    if (s.length == 4) {
        const year = parseInt(s);
        if (year && year <= 9999) {
            const components = { year };
            return [{ components, label: s }];
        }
    }
    return [];
};

const parseHolidayComponents = (s: string, holidays: DateSearchResult[]) =>
    holidays.filter(({ label }) => label.toLowerCase().includes(s));

function isMatch(file: EnteFile, query: SearchQuery) {
    if (query?.collection) {
        return query.collection === file.collectionID;
    }

    if (query?.date) {
        return isDateComponentsMatch(
            query.date,
            getUICreationDate(file, getPublicMagicMetadataSync(file)),
        );
    }
    if (query?.location) {
        return isInsideLocationTag(
            {
                latitude: file.metadata.latitude ?? null,
                longitude: file.metadata.longitude ?? null,
            },
            query.location,
        );
    }
    if (query?.city) {
        return isInsideCity(
            {
                latitude: file.metadata.latitude ?? null,
                longitude: file.metadata.longitude ?? null,
            },
            query.city,
        );
    }
    if (query?.files) {
        return query.files.indexOf(file.id) !== -1;
    }
    if (query?.person) {
        return query.person.files.indexOf(file.id) !== -1;
    }
    if (typeof query?.fileType !== "undefined") {
        return query.fileType === file.metadata.fileType;
    }
    if (typeof query?.clip !== "undefined") {
        return query.clip.has(file.id);
    }
    return false;
}

const isDateComponentsMatch = (
    { year, month, day, weekday, hour }: SearchDateComponents,
    date: Date,
) => {
    // Components are guaranteed to have at least one attribute present, so
    // start by assuming true.
    let match = true;

    if (year) match = date.getFullYear() == year;
    // JS getMonth is 0-indexed.
    if (match && month) match = date.getMonth() + 1 == month;
    if (match && day) match = date.getDate() == day;
    if (match && weekday) match = date.getDay() == weekday;
    if (match && hour) match = date.getHours() == hour;

    return match;
};

export function isInsideLocationTag(
    location: Location,
    locationTag: LocationTagData,
) {
    return isLocationCloseToPoint(
        location,
        locationTag.centerPoint,
        locationTag.radius,
    );
}

const DEFAULT_CITY_RADIUS = 10;
const KMS_PER_DEGREE = 111.16;

export function isInsideCity(location: Location, city: City) {
    return isLocationCloseToPoint(
        { latitude: city.lat, longitude: city.lng },
        location,
        DEFAULT_CITY_RADIUS,
    );
}

function isLocationCloseToPoint(
    centerPoint: Location,
    location: Location,
    radius: number,
) {
    const a = (radius * _scaleFactor(centerPoint.latitude!)) / KMS_PER_DEGREE;
    const b = radius / KMS_PER_DEGREE;
    const x = centerPoint.latitude! - location.latitude!;
    const y = centerPoint.longitude! - location.longitude!;
    if ((x * x) / (a * a) + (y * y) / (b * b) <= 1) {
        return true;
    }
    return false;
}

///The area bounded by the location tag becomes more elliptical with increase
///in the magnitude of the latitude on the caritesian plane. When latitude is
///0 degrees, the ellipse is a circle with a = b = r. When latitude incrases,
///the major axis (a) has to be scaled by the secant of the latitude.
function _scaleFactor(lat: number) {
    return 1 / Math.cos(lat * (Math.PI / 180));
}
