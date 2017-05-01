import React, { Component } from 'react';
import { Redirect } from 'react-router-dom';
import { Button, Modal } from 'react-bootstrap';
import Spinner from'react-spinkit';

const API_DOMAIN = 'http://160.39.133.251:8888';


class HomePage extends React.Component {

  constructor(props) {
    super(props);
    this.state = {
      loading: false,
      showModal: false,
      result: {}
    };
  }

  // componentDidMount() {

  // }

  // componentWillUnmount() {

  // }
  
  upload(fileEntry) {
      // !! Assumes variable fileURL contains a valid URL to a text file on the device,
      const fileURL = fileEntry.toURL();
      console.log('file url:: ' + fileURL);

      const success = function (r) {
          console.log("Successful upload...");
          console.log("Code = " + r.responseCode);
          try {
            const data = JSON.parse(r.response);
            this.setState({
              showModal: true,
              result: {
                foodName: data.food_name,
                foodDescription: data.food_description,
                image: API_DOMAIN + data.image,
              }
            });
          } catch(e) {
              alert(e);
          }
          this.setState({loading: false});
      }
      const fail = function (error) {
          alert("An error has occurred: Code = " + error.code);
          this.setState({loading: false});
      }
      const options = new FileUploadOptions();
      options.fileKey = "file";
      options.fileName = fileURL.substr(fileURL.lastIndexOf('/') + 1);
      // options.fileName = 'test';
      options.mimeType = "image/jpeg";
      options.chunkedMode = false;

      const ft = new FileTransfer();
      const apiUrl = API_DOMAIN + '/api/upload-food-image/';
      this.setState({loading: true});
      ft.upload(fileURL, encodeURI(apiUrl), success.bind(this), fail.bind(this), options);
  };

  uploadImage(imgUri) {
      console.log('image uri: ' + imgUri);
      this.getFileEntry(imgUri, this.upload.bind(this));
  }

  getFileEntry(imgUri, successCallback) {
      window.resolveLocalFileSystemURL(imgUri, function success(fileEntry) {
          console.log("got file: " + fileEntry.fullPath);
          successCallback(fileEntry);
      }, function () {
        // If don't get the FileEntry (which may happen when testing
        // on some emulators), copy to a new FileEntry.
          alert("could not get file entry ");
      });
  }

  openFilePicker() {
      const srcType = Camera.PictureSourceType.SAVEDPHOTOALBUM;
      const options = {
          // // Some common settings are 20, 50, and 100
          // quality: 50,
          destinationType: Camera.DestinationType.FILE_URI,
          // In this app, dynamically set the picture source, Camera or photo gallery
          sourceType: srcType,
          // encodingType: Camera.EncodingType.JPEG,
          // mediaType: Camera.MediaType.PICTURE,
          // allowEdit: true,
          correctOrientation: true  //Corrects Android orientation quirks
      }
      navigator.camera.getPicture(function cameraSuccess(imageUri) {
          this.uploadImage(imageUri);
      }.bind(this), function cameraError(error) {
          alert("Unable to obtain picture: " + error, "app");
      }, options);
  }

  chooseImage() {
      this.openFilePicker();
  }

  getLoadingIndicator() {
    const styles = {
      position: 'fixed',
      left: '42%',
      top: '50%',
      zIndex: '9999',
    };

    const indicator = (
      <div style={styles}>
        <Spinner spinnerName='three-bounce' />
      </div>
        );
    return indicator;
  }

  openModal() {
    this.setState({ showModal: true });
  }

  closeModal() {
    this.setState({ showModal: false });
  }

  getResultModal() {
    const data = this.state.result;
    const modal = (
      <Modal show={this.state.showModal} onHide={this.closeModal.bind(this)}>
          <Modal.Header closeButton>
            <Modal.Title>{data.foodName}</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <img src={data.image} />
            <p>{data.foodDescription}</p>
          </Modal.Body>
          <Modal.Footer>
            <Button onClick={this.closeModal.bind(this)}>Close</Button>
          </Modal.Footer>
        </Modal>
      );
    return modal;
  }

  render() {
    let loadingIndicator = null;
    if (this.state.loading) {
      loadingIndicator = this.getLoadingIndicator();
    }
    const resultModal = this.getResultModal();
    // loadingIndicator = this.getLoadingIndicator();
    const wellStyles = {margin: "auto", width: "40%"};
    return (
      <div className='homepage'>
        {loadingIndicator}
        {resultModal}
        <div style={wellStyles}>
          <Button bsStyle="primary" bsSize="large" onClick={this.chooseImage.bind(this)}>Choose Image</Button>
        </div>
      </div>
    );
  }
}

export default HomePage;