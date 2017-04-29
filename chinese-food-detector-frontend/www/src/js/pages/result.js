import React, { Component } from 'react';
import { Redirect } from 'react-router-dom'



class ResultPage extends React.Component {
    // mixins: [
    //   Router.Navigation
    // ]

    goHome() {
        // this.transitionTo('home');
    }

    render() {
        return ( 
            < div className = 'result-page' >
                result page
            < /div >
        );
    }
}


export default ResultPage;
