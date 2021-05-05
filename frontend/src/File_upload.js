import React, { useMemo, useCallback } from 'react'
import { useDropzone } from 'react-dropzone';
import './static/css/File_Upload.css'
const baseStyle = {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    padding: '20px',
    borderWidth: 2,
    borderRadius: 2,
    borderColor: '#eeeeee',
    borderStyle: 'dashed',
    backgroundColor: '#fafafa',
    color: '#bdbdbd',
    outline: 'none',
    transition: 'border .24s ease-in-out'
};

const activeStyle = {
    borderColor: '#2196f3'
};

const acceptStyle = {
    borderColor: '#00e676'
};

const rejectStyle = {
    borderColor: '#ff1744'
};

function File_upload(props) {
    const onDrop = useCallback(acceptedFiles => {
        props.setVideoFileURL(URL.createObjectURL(acceptedFiles[0]));
        props.getdata(acceptedFiles[0]);
    }, [])
    const {
        getRootProps,
        getInputProps,
        isDragActive,
        isDragAccept,
        isDragReject
    } = useDropzone({ accept: 'video/*', onDrop });

    const style = useMemo(() => ({
        ...baseStyle,
        ...(isDragActive ? activeStyle : {}),
        ...(isDragAccept ? acceptStyle : {}),
        ...(isDragReject ? rejectStyle : {})
    }), [
        isDragActive,
        isDragReject,
        isDragAccept
    ]);

    return (
        <div>
            <div className="file__upload__container" data-aos="fade-down">
                <div className="file__text">
                    Upload Your video here
                </div>

                <div {...getRootProps({ style })}>
                    <input {...getInputProps()} />
                    <p>Drag files here or click to select files</p>
                </div>
            </div>
        </div>
    )
}

export default File_upload
