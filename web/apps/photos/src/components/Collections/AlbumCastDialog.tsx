import { boxSeal } from "@/base/crypto/libsodium";
import log from "@/base/log";
import type { Collection } from "@/media/collection";
import { VerticallyCentered } from "@ente/shared/components/Container";
import DialogBoxV2 from "@ente/shared/components/DialogBoxV2";
import EnteButton from "@ente/shared/components/EnteButton";
import EnteSpinner from "@ente/shared/components/EnteSpinner";
import SingleInputForm, {
    type SingleInputFormProps,
} from "@ente/shared/components/SingleInputForm";
import castGateway from "@ente/shared/network/cast";
import { Link, Typography } from "@mui/material";
import { t } from "i18next";
import { useEffect, useState } from "react";
import { Trans } from "react-i18next";
import { v4 as uuidv4 } from "uuid";
import { loadSender } from "../../utils/useCastSender";

interface Props {
    show: boolean;
    onHide: () => void;
    currentCollection: Collection;
}

enum AlbumCastError {
    TV_NOT_FOUND = "tv_not_found",
}

declare global {
    interface Window {
        chrome: any;
    }
}

export default function AlbumCastDialog({
    show,
    onHide,
    currentCollection,
}: Props) {
    const [view, setView] = useState<
        "choose" | "auto" | "pin" | "auto-cast-error"
    >("choose");

    const [browserCanCast, setBrowserCanCast] = useState(false);
    // Make API call on component mount
    useEffect(() => {
        castGateway.revokeAllTokens();

        setBrowserCanCast(!!window.chrome);
    }, []);

    const onSubmit: SingleInputFormProps["callback"] = async (
        value,
        setFieldError,
    ) => {
        try {
            await doCast(value.trim());
            onHide();
        } catch (e) {
            const error = e as Error;
            let fieldError: string;
            switch (error.message) {
                case AlbumCastError.TV_NOT_FOUND:
                    fieldError = t("tv_not_found");
                    break;
                default:
                    fieldError = t("UNKNOWN_ERROR");
                    break;
            }

            setFieldError(fieldError);
        }
    };

    const doCast = async (pin: string) => {
        // does the TV exist? have they advertised their existence?
        const tvPublicKeyB64 = await castGateway.getPublicKey(pin);
        if (!tvPublicKeyB64) {
            throw new Error(AlbumCastError.TV_NOT_FOUND);
        }
        // generate random uuid string
        const castToken = uuidv4();

        // ok, they exist. let's give them the good stuff.
        const payload = JSON.stringify({
            castToken: castToken,
            collectionID: currentCollection.id,
            collectionKey: currentCollection.key,
        });
        const encryptedPayload = await boxSeal(btoa(payload), tvPublicKeyB64);

        // hey TV, we acknowlege you!
        await castGateway.publishCastPayload(
            pin,
            encryptedPayload,
            currentCollection.id,
            castToken,
        );
    };

    useEffect(() => {
        if (view === "auto") {
            loadSender().then(async (sender) => {
                const { cast } = sender;

                const instance = await cast.framework.CastContext.getInstance();
                try {
                    await instance.requestSession();
                } catch (e) {
                    setView("auto-cast-error");
                    log.error("Error requesting session", e);
                    return;
                }
                const session = instance.getCurrentSession();
                session.addMessageListener(
                    "urn:x-cast:pair-request",
                    (_, message) => {
                        const data = message;
                        const obj = JSON.parse(data);
                        const code = obj.code;

                        if (code) {
                            doCast(code)
                                .then(() => {
                                    setView("choose");
                                    onHide();
                                })
                                .catch((e) => {
                                    setView("auto-cast-error");
                                    log.error("Error casting to TV", e);
                                });
                        }
                    },
                );

                const collectionID = currentCollection.id;
                session
                    .sendMessage("urn:x-cast:pair-request", { collectionID })
                    .then(() => {
                        log.debug(() => "Message sent successfully");
                    })
                    .catch((e) => {
                        log.error("Error sending message", e);
                    });
            });
        }
    }, [view]);

    useEffect(() => {
        if (show) {
            castGateway.revokeAllTokens();
        }
    }, [show]);

    return (
        <DialogBoxV2
            sx={{ zIndex: 1600 }}
            open={show}
            onClose={onHide}
            attributes={{
                title: t("cast_album_to_tv"),
            }}
        >
            {view === "choose" && (
                <>
                    {browserCanCast && (
                        <>
                            <Typography color={"text.muted"}>
                                {t("cast_auto_pair_description")}
                            </Typography>

                            <EnteButton
                                style={{
                                    marginBottom: "1rem",
                                }}
                                onClick={() => {
                                    setView("auto");
                                }}
                            >
                                {t("cast_auto_pair")}
                            </EnteButton>
                        </>
                    )}
                    <Typography color="text.muted">
                        {t("pair_with_pin_description")}
                    </Typography>

                    <EnteButton
                        onClick={() => {
                            setView("pin");
                        }}
                    >
                        {t("pair_with_pin")}
                    </EnteButton>
                </>
            )}
            {view === "auto" && (
                <VerticallyCentered gap="1rem">
                    <EnteSpinner />
                    <Typography>{t("choose_device_from_browser")}</Typography>
                    <EnteButton
                        variant="text"
                        onClick={() => {
                            setView("choose");
                        }}
                    >
                        {t("GO_BACK")}
                    </EnteButton>
                </VerticallyCentered>
            )}
            {view === "auto-cast-error" && (
                <VerticallyCentered gap="1rem">
                    <Typography>{t("cast_auto_pair_failed")}</Typography>
                    <EnteButton
                        variant="text"
                        onClick={() => {
                            setView("choose");
                        }}
                    >
                        {t("GO_BACK")}
                    </EnteButton>
                </VerticallyCentered>
            )}
            {view === "pin" && (
                <>
                    <Typography>
                        <Trans
                            i18nKey="visit_cast_url"
                            components={{
                                a: (
                                    <Link
                                        target="_blank"
                                        href="https://cast.ente.io"
                                    />
                                ),
                            }}
                            values={{ url: "cast.ente.io" }}
                        />
                    </Typography>
                    <Typography>{t("enter_cast_pin_code")}</Typography>
                    <SingleInputForm
                        callback={onSubmit}
                        fieldType="text"
                        realLabel={"Code"}
                        realPlaceholder={"123456"}
                        buttonText={t("pair_device_to_tv")}
                        submitButtonProps={{ sx: { mt: 1, mb: 2 } }}
                    />
                    <EnteButton
                        variant="text"
                        onClick={() => {
                            setView("choose");
                        }}
                    >
                        {t("GO_BACK")}
                    </EnteButton>
                </>
            )}
        </DialogBoxV2>
    );
}
