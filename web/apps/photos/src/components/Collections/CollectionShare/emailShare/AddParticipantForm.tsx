import {
    MenuItemDivider,
    MenuItemGroup,
    MenuSectionTitle,
} from "@/base/components/Menu";
import { FocusVisibleButton } from "@/base/components/mui/FocusVisibleButton";
import { LoadingButton } from "@/base/components/mui/LoadingButton";
import { FlexWrapper } from "@ente/shared/components/Container";
import { EnteMenuItem } from "@ente/shared/components/Menu/EnteMenuItem";
import DoneIcon from "@mui/icons-material/Done";
import { FormHelperText, Stack } from "@mui/material";
import TextField from "@mui/material/TextField";
import Avatar from "components/pages/gallery/Avatar";
import { Formik, type FormikHelpers } from "formik";
import { t } from "i18next";
import { useMemo, useState } from "react";
import * as Yup from "yup";

interface formValues {
    inputValue: string;
    selectedOptions: string[];
}
export interface AddParticipantFormProps {
    callback: (props: { email?: string; emails?: string[] }) => Promise<void>;
    fieldType: "text" | "email" | "password";
    placeholder: string;
    buttonText: string;
    submitButtonProps?: any;
    initialValue?: string;
    secondaryButtonAction?: () => void;
    disableAutoFocus?: boolean;
    hiddenPreInput?: any;
    caption?: any;
    hiddenPostInput?: any;
    autoComplete?: string;
    blockButton?: boolean;
    hiddenLabel?: boolean;
    onClose?: () => void;
    optionsList?: string[];
}

export default function AddParticipantForm(props: AddParticipantFormProps) {
    const { submitButtonProps } = props;
    const { sx: buttonSx, ...restSubmitButtonProps } = submitButtonProps ?? {};

    const [loading, SetLoading] = useState(false);

    const submitForm = async (
        values: formValues,
        { setFieldError, resetForm }: FormikHelpers<formValues>,
    ) => {
        try {
            SetLoading(true);
            if (values.inputValue !== "") {
                await props.callback({ email: values.inputValue });
            } else if (values.selectedOptions.length !== 0) {
                await props.callback({ emails: values.selectedOptions });
            }
            SetLoading(false);
            props.onClose();
            resetForm();
        } catch (e) {
            setFieldError("inputValue", e?.message);
            SetLoading(false);
        }
    };

    const validationSchema = useMemo(() => {
        switch (props.fieldType) {
            case "text":
                return Yup.object().shape({
                    inputValue: Yup.string().required(t("required")),
                });
            case "email":
                return Yup.object().shape({
                    inputValue: Yup.string().email(t("EMAIL_ERROR")),
                });
        }
    }, [props.fieldType]);

    const handleInputFieldClick = (setFieldValue) => {
        setFieldValue("selectedOptions", []);
    };

    return (
        <Formik<formValues>
            initialValues={{
                inputValue: props.initialValue ?? "",
                selectedOptions: [],
            }}
            onSubmit={submitForm}
            validationSchema={validationSchema}
            validateOnChange={false}
            validateOnBlur={false}
        >
            {({
                values,
                errors,
                handleChange,
                handleSubmit,
                setFieldValue,
            }) => (
                <form noValidate onSubmit={handleSubmit}>
                    <Stack spacing={"24px"} py={"20px"} px={"12px"}>
                        {props.hiddenPreInput}
                        <Stack>
                            <MenuSectionTitle title={t("ADD_NEW_EMAIL")} />
                            <TextField
                                sx={{ marginTop: 0 }}
                                hiddenLabel={props.hiddenLabel}
                                fullWidth
                                type={props.fieldType}
                                id={props.fieldType}
                                onChange={handleChange("inputValue")}
                                onClick={() =>
                                    handleInputFieldClick(setFieldValue)
                                }
                                name={props.fieldType}
                                {...(props.hiddenLabel
                                    ? { placeholder: props.placeholder }
                                    : { label: props.placeholder })}
                                error={Boolean(errors.inputValue)}
                                helperText={errors.inputValue}
                                value={values.inputValue}
                                disabled={loading}
                                autoFocus={!props.disableAutoFocus}
                                autoComplete={props.autoComplete}
                            />
                        </Stack>

                        {props.optionsList.length > 0 && (
                            <Stack>
                                <MenuSectionTitle
                                    title={t("OR_ADD_EXISTING")}
                                />
                                <MenuItemGroup>
                                    {props.optionsList.map((item, index) => (
                                        <>
                                            <EnteMenuItem
                                                fontWeight="normal"
                                                key={item}
                                                onClick={() => {
                                                    if (
                                                        values.selectedOptions.includes(
                                                            item,
                                                        )
                                                    ) {
                                                        setFieldValue(
                                                            "selectedOptions",
                                                            values.selectedOptions.filter(
                                                                (
                                                                    selectedOption,
                                                                ) =>
                                                                    selectedOption !==
                                                                    item,
                                                            ),
                                                        );
                                                    } else {
                                                        setFieldValue(
                                                            "selectedOptions",
                                                            [
                                                                ...values.selectedOptions,
                                                                item,
                                                            ],
                                                        );
                                                    }
                                                }}
                                                label={item}
                                                startIcon={
                                                    <Avatar email={item} />
                                                }
                                                endIcon={
                                                    values.selectedOptions.includes(
                                                        item,
                                                    ) ? (
                                                        <DoneIcon />
                                                    ) : null
                                                }
                                            />
                                            {index !==
                                                props.optionsList.length -
                                                    1 && <MenuItemDivider />}
                                        </>
                                    ))}
                                </MenuItemGroup>
                            </Stack>
                        )}

                        <FormHelperText
                            sx={{
                                position: "relative",
                                top: errors.inputValue ? "-22px" : "0",
                                float: "right",
                                padding: "0 8px",
                            }}
                        >
                            {props.caption}
                        </FormHelperText>
                        {props.hiddenPostInput}
                    </Stack>
                    <FlexWrapper
                        px={"8px"}
                        justifyContent={"center"}
                        flexWrap={props.blockButton ? "wrap-reverse" : "nowrap"}
                    >
                        <Stack direction={"column"} px={"8px"} width={"100%"}>
                            {props.secondaryButtonAction && (
                                <FocusVisibleButton
                                    onClick={props.secondaryButtonAction}
                                    size="large"
                                    color="secondary"
                                    sx={{
                                        "&&&": {
                                            mt: !props.blockButton ? 2 : 0.5,
                                            mb: !props.blockButton ? 4 : 0,
                                            mr: !props.blockButton ? 1 : 0,
                                            ...buttonSx,
                                        },
                                    }}
                                    {...restSubmitButtonProps}
                                >
                                    {t("cancel")}
                                </FocusVisibleButton>
                            )}

                            <LoadingButton
                                type="submit"
                                color="accent"
                                fullWidth
                                buttonText={props.buttonText}
                                loading={loading}
                                sx={{ mt: 2, mb: 4 }}
                                {...restSubmitButtonProps}
                            >
                                {props.buttonText}
                            </LoadingButton>
                        </Stack>
                    </FlexWrapper>
                </form>
            )}
        </Formik>
    );
}
