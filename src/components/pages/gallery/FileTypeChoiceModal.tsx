import { Modal, Button } from 'react-bootstrap';
import React from 'react';
import ImportService from 'services/importService';
import { getElectronFiles } from 'utils/upload';

export default function FileTypeChoiceModal({
    setElectronFiles,
    showFiletypeModal,
    setShowFiletypeModal,
}) {
    const hideFiletypeDialog = () => {
        setShowFiletypeModal(false);
    };

    const uploadFiles = async () => {
        const filePaths = await ImportService.showUploadFilesDialog();
        hideFiletypeDialog();
        const files = await getElectronFiles(filePaths);
        setElectronFiles(files);
    };

    const uploadDirs = async () => {
        const filePaths = await ImportService.showUploadDirsDialog();
        hideFiletypeDialog();
        const files = await getElectronFiles(filePaths);
        setElectronFiles(files);
    };

    return (
        <Modal
            show={showFiletypeModal}
            aria-labelledby="contained-modal-title-vcenter"
            centered>
            <Modal.Header closeButton onHide={hideFiletypeDialog}>
                <Modal.Title id="contained-modal-title-vcenter">
                    Upload
                </Modal.Title>
            </Modal.Header>
            <Modal.Body
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-around',
                    height: '12vh',
                }}>
                <Button variant="outline-success" onClick={uploadFiles}>
                    Upload Files
                </Button>
                <Button variant="outline-success" onClick={uploadDirs}>
                    Upload Folders
                </Button>
            </Modal.Body>
        </Modal>
    );
}
