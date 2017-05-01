import React, { Component } from 'react';
import { Redirect } from 'react-router-dom'



class AboutPage extends React.Component {
    // mixins: [
    //   Router.Navigation
    // ]

    goHome() {
        // this.transitionTo('home');
    }

    render() {
        return ( 
            < div className = 'about-page' >
                about page
            < /div >
        );
    }
}


export default AboutPage;
